//
// Created by Mgepahmge on 2025/12/19.
//
#include <algorithm>
#include <unordered_map>
#include <vector>
#include "SpMSpV.cuh"
#include "src/Query/check.cuh"
#include "src/utils/NVTXProfiler.cuh"


/**
 * @brief Multiply a sparse tensor in CSC format with a grid represented as a sparse matrix.
 *
 * @param[in] P Pointer to the sparse tensor in CSC format.(device/host)
 * @param[in] grid Pointer to the grid represented as a sparse matrix.(device)
 *
 * @return GridAsSparseMatrix The resulting grid after multiplication.(device)
 */
GridAsSparseMatrix* SpTSpMMultiplication_v3(SparseTensorCscFormat* P, GridAsSparseMatrix* grid, double* d_P_values) {
    NvtxProfiler profiler("SpTSpMMultiplication_v3", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Orange);

    const auto numDims = grid->get_dimensions();
    const auto P_row_nums = P->get_row_nums();
    const auto grid_size = grid->get_nnz_nums();

    // P struct metadata is always on host -- use directly, no download needed.
    const size_t* h_matrixColPtr = P->get_col_res(); // [HOST]
    const size_t* h_matrixRowInd = P->get_row_ids(); // [HOST]
    double* d_matrixData = d_P_values; // [DEVICE] -- pre-uploaded

    // Download grid index from device (needed for CPU-side index scatter).
    auto* h_vectorIndex = new size_t[grid_size];
    CUDA_CHECK(cudaMemcpy(h_vectorIndex, grid->get_ids_(), grid_size * sizeof(size_t), cudaMemcpyDeviceToHost));
    auto* d_vectorData = const_cast<double*>(grid->get_vals_()); // [DEVICE]

    // Step 1: count output non-zeros
    NvtxProfiler p1("sptrsm_count", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Rose);
    unsigned int numProcessedNonZero = 0;
    for (size_t i = 0; i < grid_size; ++i) {
        const auto col = h_vectorIndex[i];
        numProcessedNonZero += static_cast<unsigned int>(h_matrixColPtr[col + 1] - h_matrixColPtr[col]);
    }
    p1.release();

    // Step 2: allocate index arrays
    NvtxProfiler p2("sptrsm_alloc", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::LimeGreen);
    auto* h_processedColInd = new size_t[numProcessedNonZero];
    auto* h_processedRowInd = new size_t[numProcessedNonZero];
    auto* h_processedMatrixPos = new size_t[numProcessedNonZero];
    size_t* d_processedColInd = nullptr;
    size_t* d_processedRowInd = nullptr;
    size_t* d_processedMatrixPos = nullptr;
    CUDA_CHECK(cudaMalloc(&d_processedColInd, numProcessedNonZero * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_processedRowInd, numProcessedNonZero * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_processedMatrixPos, numProcessedNonZero * sizeof(size_t)));
    p2.release();

    // Step 3: scatter indices on host
    NvtxProfiler p3("sptrsm_scatter", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Magenta);
    unsigned int writeIdx = 0;
    for (size_t i = 0; i < grid_size; ++i) {
        const auto col = h_vectorIndex[i];
        const auto colStart = h_matrixColPtr[col];
        const auto colEnd = h_matrixColPtr[col + 1];
        for (auto j = colStart; j < colEnd; ++j) {
            h_processedColInd[writeIdx] = i;
            h_processedRowInd[writeIdx] = h_matrixRowInd[j];
            h_processedMatrixPos[writeIdx] = j;
            ++writeIdx;
        }
    }
    p3.release();

    // Upload index arrays to device
    CUDA_CHECK(
        cudaMemcpy(d_processedColInd, h_processedColInd, numProcessedNonZero * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_processedRowInd, h_processedRowInd, numProcessedNonZero * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_processedMatrixPos, h_processedMatrixPos, numProcessedNonZero * sizeof(size_t),
                          cudaMemcpyHostToDevice));

    // Step 4: launch kernel
    const auto totalNumNoneZero = static_cast<unsigned int>(numProcessedNonZero) * numDims;
    double* yValue = nullptr;
    CUDA_CHECK(cudaMalloc(&yValue, totalNumNoneZero * sizeof(double)));

    const dim3 block(512);
    const dim3 grid_dim((totalNumNoneZero + UNROLL_FACTOR * 512 - 1) / (UNROLL_FACTOR * 512));
    SpMSpVKernelAOS_v2<<<grid_dim, block>>>(yValue, d_processedColInd, d_processedMatrixPos, d_matrixData, d_vectorData,
                                            numDims, totalNumNoneZero);
    size_t* yIndex = d_processedRowInd;

    // Step 5: cleanup host and temporary device buffers
    delete[] h_vectorIndex;
    delete[] h_processedColInd;
    delete[] h_processedRowInd;
    delete[] h_processedMatrixPos;

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_processedColInd));
    CUDA_CHECK(cudaFree(d_processedMatrixPos));

    return new GridAsSparseMatrix{P_row_nums, numDims, numProcessedNonZero, yIndex, yValue};
}

GridAsSparseMatrix* SpTSpMMultiplication_v3_L2(SparseTensorCscFormat* P, GridAsSparseMatrix* grid) {
    NvtxProfiler profiler("SpTSpMMultiplication_v3_L2", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Orange);

    const auto numDims = grid->get_dimensions();
    const auto P_row_nums = P->get_row_nums();
    const auto grid_size = grid->get_nnz_nums();
    const size_t* h_colPtr = P->get_col_res(); // [HOST]
    const size_t* h_rowInd = P->get_row_ids(); // [HOST]
    const double* h_pVals = P->get_vals(); // [HOST]

    // Step 1: download pruned centroid ids from device
    auto* h_vectorIndex = new size_t[grid_size];
    CUDA_CHECK(cudaMemcpy(h_vectorIndex, grid->get_ids_(), grid_size * sizeof(size_t), cudaMemcpyDeviceToHost));

    // Step 2: count output non-zeros
    unsigned int numProcessedNonZero = 0;
    for (size_t i = 0; i < grid_size; ++i) {
        const auto col = h_vectorIndex[i];
        numProcessedNonZero += static_cast<unsigned int>(h_colPtr[col + 1] - h_colPtr[col]);
    }

    // Step 3: build compact host arrays
    //   hPValsCompact: only the needed column vals (numProcessedNonZero * D)
    //   hColInd: local centroid index per nnz
    //   hRowInd: output fine-point id per nnz
    //   hMatPos: local nnz index (= writeIdx, for dPValsCompact access)
    auto* hPValsCompact = new double[numProcessedNonZero * numDims];
    auto* hColInd = new size_t[numProcessedNonZero];
    auto* hRowInd = new size_t[numProcessedNonZero];
    auto* hMatPos = new size_t[numProcessedNonZero];

    unsigned int writeIdx = 0;
    for (size_t i = 0; i < grid_size; ++i) {
        const auto col = h_vectorIndex[i];
        const auto colStart = h_colPtr[col];
        const auto colEnd = h_colPtr[col + 1];
        const auto colNnz = colEnd - colStart;

        // Collect P vals for this column (contiguous block -- one memcpy)
        std::memcpy(hPValsCompact + writeIdx * numDims, h_pVals + colStart * numDims,
                    colNnz * numDims * sizeof(double));

        for (auto j = colStart; j < colEnd; ++j) {
            hColInd[writeIdx] = i;
            hRowInd[writeIdx] = h_rowInd[j];
            hMatPos[writeIdx] = writeIdx; // local offset into dPValsCompact
            ++writeIdx;
        }
    }

    delete[] h_vectorIndex;

    // Step 4: upload compact P vals + index arrays to device
    double* dPValsCompact = nullptr;
    size_t* dColInd = nullptr;
    size_t* dRowInd = nullptr;
    size_t* dMatPos = nullptr;
    CUDA_CHECK(cudaMalloc(&dPValsCompact, (size_t)numProcessedNonZero * numDims * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dColInd, numProcessedNonZero * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&dRowInd, numProcessedNonZero * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&dMatPos, numProcessedNonZero * sizeof(size_t)));

    CUDA_CHECK(cudaMemcpy(dPValsCompact, hPValsCompact, (size_t)numProcessedNonZero * numDims * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dColInd, hColInd, numProcessedNonZero * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dRowInd, hRowInd, numProcessedNonZero * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dMatPos, hMatPos, numProcessedNonZero * sizeof(size_t), cudaMemcpyHostToDevice));

    delete[] hPValsCompact;
    delete[] hColInd;
    delete[] hRowInd;
    delete[] hMatPos;

    // Step 5: launch kernel
    double* dVectorData = const_cast<double*>(grid->get_vals_()); // [DEVICE]
    const auto totalNumNoneZero = static_cast<unsigned int>(numProcessedNonZero) * numDims;
    double* yValue = nullptr;
    CUDA_CHECK(cudaMalloc(&yValue, totalNumNoneZero * sizeof(double)));

    const dim3 block(512);
    const dim3 grid_dim((totalNumNoneZero + UNROLL_FACTOR * 512 - 1) / (UNROLL_FACTOR * 512));
    SpMSpVKernelAOS_v2<<<grid_dim, block>>>(yValue, dColInd, dMatPos, dPValsCompact, dVectorData, numDims,
                                            totalNumNoneZero);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup: free compact P vals, colInd, matPos; keep rowInd as yIndex
    CUDA_CHECK(cudaFree(dPValsCompact));
    CUDA_CHECK(cudaFree(dColInd));
    CUDA_CHECK(cudaFree(dMatPos));

    return new GridAsSparseMatrix{P_row_nums, numDims, numProcessedNonZero, dRowInd, yValue};
}

GridAsSparseMatrix* SpTSpMMultiplication_v3_L0_nb(SparseTensorCscFormat* P, GridAsSparseMatrix* grid,
                                                  double* d_P_values, size_t* dColIndBuf, size_t* dRowIndBuf,
                                                  size_t* dMatrixPosBuf, double* dYValueBuf, size_t* hVectorIndex,
                                                  size_t* hProcColInd, size_t* hProcRowInd, size_t* hProcMatrixPos) {
    NvtxProfiler profiler("SpTSpMMultiplication_v3_L0_nb", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Orange);

    const auto numDims = grid->get_dimensions();
    const auto P_row_nums = P->get_row_nums();
    const auto grid_size = grid->get_nnz_nums();

    const size_t* h_matrixColPtr = P->get_col_res();
    const size_t* h_matrixRowInd = P->get_row_ids();
    double* d_vectorData = const_cast<double*>(grid->get_vals_());

    // Step 1: download grid ids into pre-allocated host buffer (no new[])
    CUDA_CHECK(cudaMemcpy(hVectorIndex, grid->get_ids_(), grid_size * sizeof(size_t), cudaMemcpyDeviceToHost));

    // Step 2: count output non-zeros
    unsigned int numProcessedNonZero = 0;
    for (size_t i = 0; i < grid_size; ++i) {
        const auto col = hVectorIndex[i];
        numProcessedNonZero += static_cast<unsigned int>(h_matrixColPtr[col + 1] - h_matrixColPtr[col]);
    }

    // Step 3: build index arrays into pre-allocated host staging (no new[])
    unsigned int writeIdx = 0;
    for (size_t i = 0; i < grid_size; ++i) {
        const auto col = hVectorIndex[i];
        const auto colStart = h_matrixColPtr[col];
        const auto colEnd = h_matrixColPtr[col + 1];
        for (auto j = colStart; j < colEnd; ++j) {
            hProcColInd[writeIdx] = i;
            hProcRowInd[writeIdx] = h_matrixRowInd[j];
            hProcMatrixPos[writeIdx] = j;
            ++writeIdx;
        }
    }

    // Step 4: upload index arrays into pre-allocated device buffers (no cudaMalloc)
    CUDA_CHECK(cudaMemcpy(dColIndBuf, hProcColInd, numProcessedNonZero * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dRowIndBuf, hProcRowInd, numProcessedNonZero * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dMatrixPosBuf, hProcMatrixPos, numProcessedNonZero * sizeof(size_t), cudaMemcpyHostToDevice));

    // Step 5: launch kernel using pre-allocated yValue buffer (no cudaMalloc)
    const auto totalNumNoneZero = static_cast<unsigned int>(numProcessedNonZero) * numDims;
    const dim3 block(512);
    const dim3 grid_dim((totalNumNoneZero + UNROLL_FACTOR * 512 - 1) / (UNROLL_FACTOR * 512));
    SpMSpVKernelAOS_v2<<<grid_dim, block>>>(dYValueBuf, dColIndBuf, dMatrixPosBuf, d_P_values, d_vectorData, numDims,
                                            totalNumNoneZero);
    CUDA_CHECK(cudaDeviceSynchronize());

    // dRowIndBuf reused as yIndex; dYValueBuf is output vals.
    // Both owned by IndexData -- do NOT free here.
    return new GridAsSparseMatrix{P_row_nums, numDims, numProcessedNonZero, dRowIndBuf, dYValueBuf};
}
