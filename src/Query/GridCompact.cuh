/**
 * @file GridCompact.cuh
 * @brief Compaction kernel and wrapper for SparseGrid pruning output.
 *
 * @details Provides a warp-level prefix-sum scatter kernel (gridCompactPrefixKernel)
 * and the compactGrid wrapper that removes all entries where mask is false
 * from a device-resident SparseGrid.  Used by L1 and L2 strategies.
 * L0 calls the low-level kernels directly. L3 uses its own tiled compact path.
 */

#pragma once

#include <cstddef>
#include "src/Data_Structures/Data_Structures.cuh"

/*!
 * @brief Warp-level prefix-sum scatter kernel for SparseGrid compaction.
 *
 * @details Each warp processes Iter * 32 elements.  Within a warp, a
 * prefix sum over the mask selects the output positions.  The per-warp
 * output base is taken from processorCounts[warpIdx - 1], which must
 * contain the inclusive prefix sum of per-warp selected counts (computed
 * by launchCountKernel + launchPrefixSumKernel before this kernel runs).
 *
 * The template parameter Iter equals kKp / 32, where kKp is the number
 * of elements processed per warp (128 by default, giving Iter = 4).
 *
 * @tparam Iter  Number of loop iterations per warp (kKp / 32).
 * @param[out] outputData       Compacted vals array (device).
 * @param[out] outputIndex      Compacted ids array (device).
 * @param[in]  inputData        Input vals array (device).
 * @param[in]  inputIndex       Input ids array (device).
 * @param[in]  mask             Selection mask (device, dataSize bools).
 * @param[in]  processorCounts  Inclusive prefix-sum of per-warp counts (device).
 * @param[in]  dataSize         Total number of input elements.
 * @param[in]  dim              Data dimensionality.
 * @param[in]  nProcessors      Number of warps (== ceil(dataSize / kKp)).
 * @param[in]  validCount       Total number of selected elements.
 * @note This kernel must not be modified.
 */
template <int Iter>
__global__ void gridCompactPrefixKernel(double* outputData, size_t* outputIndex, const double* inputData,
                                        const size_t* inputIndex, const bool* mask, const unsigned int* processorCounts,
                                        const unsigned int dataSize, const size_t dim, const size_t nProcessors,
                                        const size_t validCount) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto warpIdx = idx >> 5;
    if (warpIdx >= nProcessors) {
        return;
    }

    const auto laneIdx = idx & 31;
    const auto begin = warpIdx * (Iter << 5);
    auto outputBegin = warpIdx > 0 ? processorCounts[warpIdx - 1] : 0;

#pragma unroll
    for (auto i = 0; i < Iter; ++i) {
        const auto current = begin + (i << 5) + laneIdx;
        unsigned int localFlags = 0;
        bool localMask = false;
        if (current < dataSize) {
            localMask = mask[current];
            if (localMask) {
                localFlags = 1;
            }
        }
        unsigned int temp;
        temp = __shfl_up_sync(0xFFFFFFFF, localFlags, 1);
        if (laneIdx >= 1) {
            localFlags += temp;
        }
        temp = __shfl_up_sync(0xFFFFFFFF, localFlags, 2);
        if (laneIdx >= 2) {
            localFlags += temp;
        }
        temp = __shfl_up_sync(0xFFFFFFFF, localFlags, 4);
        if (laneIdx >= 4) {
            localFlags += temp;
        }
        temp = __shfl_up_sync(0xFFFFFFFF, localFlags, 8);
        if (laneIdx >= 8) {
            localFlags += temp;
        }
        temp = __shfl_up_sync(0xFFFFFFFF, localFlags, 16);
        if (laneIdx >= 16) {
            localFlags += temp;
        }

        if (localMask) {
            const auto outIdx = outputBegin + localFlags - 1;
            if (outIdx < validCount && current < dataSize) {
                outputIndex[outIdx] = inputIndex[current];
                for (size_t d = 0; d < dim; ++d) {
                    outputData[outIdx * dim + d] = inputData[current * dim + d];
                }
            }
        }
        outputBegin += __shfl_sync(0xFFFFFFFF, localFlags, 31);
    }
}

/*!
 * @brief Retain only the entries of grid where mask is true.
 *
 * @details Internally runs three steps: count selected per warp, prefix-sum
 * the counts, then scatter selected entries to the output via
 * gridCompactPrefixKernel.  The mask is uploaded from host to device
 * internally; all other buffers are allocated on device inside this function.
 *
 * @param[in] grid        Input SparseGrid; ids and vals must reside in device memory.
 * @param[in] mask        Host selection mask of length grid.numNnz().
 * @param[in] validCount  Number of true entries in mask.
 * @return New SparseGrid whose ids and vals are device-allocated.
 *         The caller must free via cudaFree(ids), cudaFree(vals), delete.
 */
SparseGrid* compactGrid(const SparseGrid& grid, const bool* mask, size_t validCount);
