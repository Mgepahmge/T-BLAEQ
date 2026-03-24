/**
 * @file SpMSpV.cuh
 * @brief SpMSpV kernels and strategy-specific wrappers for the T-BLAEQ query pipeline.
 *
 * @details This file provides the low-level CUDA kernels for sparse matrix times
 * sparse vector (SpMSpV) multiplication in AOS and SOA memory layouts, along with
 * the legacy AOS/SOA handle classes and the four strategy-specific wrappers used
 * by L0, L1, L2, and L3:
 *
 * SpTSpMMultiplication_v3       - standard path (L1): P vals pre-uploaded, dynamic alloc.
 * SpTSpMMultiplication_v3_L2   - L2 path: only selected P columns uploaded per query.
 * SpTSpMMultiplication_v3_L0_nb - L0 path: zero cudaMalloc via pre-allocated buffers.
 *
 * L3 manages SpTSpM entirely within L3Strategy::runSpTSpMTiled() using the
 * low-level kernels directly; it does not have a dedicated wrapper here.
 */

#ifndef SPMSPV_SPMSPV_CUH
#define SPMSPV_SPMSPV_CUH

#include <cstring>
#include "src/Data_Structures/Data_Structures.cuh"

constexpr unsigned int UNROLL_FACTOR = 4; //!< Unroll factor for SpMSpVKernelAOS and SpMSpVKernelAOS_v2.

/*!
 * @brief SpMSpV kernel v2: AOS layout with per-nnz matrix position index.
 *
 * @details Differs from SpMSpVKernelAOS in that it accepts a matrixPosInd array
 * that holds the local offset into matrixData for each nnz element.  This allows
 * matrixData to be a compact subset of the full P-tensor (as produced by the L2
 * and L0 wrappers), rather than the full P-tensor array starting at offset 0.
 *
 * yValue[i] = matrixData[matrixPosInd[i/numDims] * numDims + i % numDims]
 *           * xValue[colInd[i/numDims] * numDims + i % numDims]
 *
 * @tparam Integer Index type.
 * @tparam Real    Floating-point type.
 * @param[out] yValue           Output values (totalNumNoneZero Reals, device).
 * @param[in]  colInd           Per-nnz local centroid index into xValue (device).
 * @param[in]  matrixPosInd     Per-nnz local offset into matrixData (device).
 * @param[in]  matrixData       Compact P-tensor values in AOS layout (device).
 * @param[in]  xValue           Input sparse vector values in AOS layout (device).
 * @param[in]  numDims          Data dimensionality.
 * @param[in]  totalNumNoneZero Total number of scalar outputs (numNnz * numDims).
 */
template <typename Integer, typename Real>
__global__ void SpMSpVKernelAOS_v2(Real* yValue, const Integer* colInd, const Integer* matrixPosInd,
                                   const Real* matrixData, const Real* xValue, const unsigned int numDims,
                                   const unsigned int totalNumNoneZero) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = blockDim.x * gridDim.x;

#pragma unroll
    for (auto i = 0; i < UNROLL_FACTOR; ++i) {
        if (const auto index = idx + i * stride; index < totalNumNoneZero) {
            const auto elementIdx = index / numDims; // 当前元素索引
            const auto dim = index % numDims; // 当前维度

            const auto col = colInd[elementIdx]; // 输入向量的列
            const auto matPos = matrixPosInd[elementIdx]; // 原始矩阵中的位置

            // 直接从原始矩阵读取数据并计算
            yValue[index] = matrixData[matPos * numDims + dim] * xValue[col * numDims + dim];
        }
    }
}


/*!
 * @brief Standard SpTSpM wrapper used by L1Strategy.
 *
 * @details Downloads grid ids to host, builds index arrays for the selected
 * P columns, uploads them to device, launches SpMSpVKernelAOS_v2, and
 * returns a new device-resident SparseGrid.  Dynamically allocates all
 * working arrays and the output yValue buffer.
 *
 * @param[in]     P           P-tensor in CSC format (host metadata, device vals).
 * @param[in]     grid        Device-resident input SparseGrid (pruned centroids).
 * @param[in]     d_P_values  Pre-uploaded P-tensor values on the device.
 * @return New SparseGrid with device-allocated ids (rowInd) and vals (yValue).
 *         Caller owns both; free with cudaFree and delete.
 */
GridAsSparseMatrix* SpTSpMMultiplication_v3(SparseTensorCscFormat* P, GridAsSparseMatrix* grid, double* d_P_values);

/*!
 * @brief L2 SpTSpM wrapper: selectively uploads only the needed P columns.
 *
 * @details Downloads pruned centroid ids, collects the corresponding P-tensor
 * columns from host memory into a compact buffer using memcpy per column
 * (avoiding element-by-element copies), uploads the compact buffer, and
 * launches SpMSpVKernelAOS_v2.  matrixPosInd holds local offsets into the
 * compact buffer rather than global P-tensor offsets.
 *
 * This avoids uploading the full P-tensor array when selectCount << col_nums,
 * which is typical for KNN queries, dramatically reducing transfer overhead.
 *
 * @param[in] P    P-tensor in CSC format; vals remain on host throughout.
 * @param[in] grid Device-resident input SparseGrid (pruned centroids).
 * @return New SparseGrid with device-allocated ids and vals. Caller owns both.
 */
GridAsSparseMatrix* SpTSpMMultiplication_v3_L2(SparseTensorCscFormat* P, GridAsSparseMatrix* grid);

/*!
 * @brief L0 SpTSpM wrapper: zero cudaMalloc in the hot path.
 *
 * @details Identical logic to SpTSpMMultiplication_v3 but uses caller-supplied
 * pre-allocated buffers for all working arrays and the output yValue, eliminating
 * every cudaMalloc call from the query hot path.
 *
 * The returned SparseGrid's ids_ points to dRowIndBuf and vals_ points to
 * dYValueBuf; both are owned by IndexData.  The caller must set
 * LevelResult::ownsIds = false and ownsVals = false accordingly.
 *
 * @param[in]  P              P-tensor in CSC format.
 * @param[in]  grid           Device-resident input SparseGrid.
 * @param[in]  d_P_values     Pre-uploaded P-tensor values (device).
 * @param[in]  dColIndBuf     Pre-allocated column index buffer (device, >= P.row_nums).
 * @param[in]  dRowIndBuf     Pre-allocated row index buffer (device); reused as yIndex.
 * @param[in]  dMatrixPosBuf  Pre-allocated matrix position buffer (device, >= P.row_nums).
 * @param[in]  dYValueBuf     Pre-allocated output value buffer (device, >= P.nnz * D).
 * @param[in]  hVectorIndex   Pre-allocated host staging for grid ids (>= grid.nnz).
 * @param[in]  hProcColInd    Pre-allocated host staging for column indices (>= P.nnz).
 * @param[in]  hProcRowInd    Pre-allocated host staging for row indices (>= P.nnz).
 * @param[in]  hProcMatrixPos Pre-allocated host staging for matrix positions (>= P.nnz).
 * @return SparseGrid view backed by dRowIndBuf and dYValueBuf; caller does not own.
 */
GridAsSparseMatrix* SpTSpMMultiplication_v3_L0_nb(SparseTensorCscFormat* P, GridAsSparseMatrix* grid,
                                                  double* d_P_values, size_t* dColIndBuf, size_t* dRowIndBuf,
                                                  size_t* dMatrixPosBuf, double* dYValueBuf, size_t* hVectorIndex,
                                                  size_t* hProcColInd, size_t* hProcRowInd, size_t* hProcMatrixPos);

#endif // SPMSPV_SPMSPV_CUH
