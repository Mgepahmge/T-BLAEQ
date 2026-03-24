/**
 * @file compact.cuh
 * @brief Low-level CUDA kernels and launchers for warp-parallel stream compaction.
 *
 * @details This file provides the building blocks for GPU-accelerated stream
 * compaction: a warp-level count kernel, a two-pass block-level inclusive
 * prefix-sum, a scatter kernel, and a legacy CompactHandler class.
 *
 * The three-step pipeline used by GridCompact and L0/L3 strategies is:
 *   1. launchCountKernel   - count selected elements per warp.
 *   2. launchPrefixSumKernel - compute inclusive prefix-sum of warp counts.
 *   3. gridCompactPrefixKernel (GridCompact.cuh) or compactPrefixKernel -
 *      scatter selected elements to the output using the prefix-sum offsets.
 */

#ifndef CUDADB_COMPACT_CUH
#define CUDADB_COMPACT_CUH

#include "check.cuh"

/*!
 * @brief Count the number of non-zero (positive) elements per warp.
 *
 * @details Each warp processes Iter * 32 consecutive elements.  Counts are
 * accumulated using warp-shuffle reduction and written to processorCounts
 * at the warp index.  The result feeds launchPrefixSumKernel.
 *
 * @tparam T    Element type; elements with value > 0 are counted as selected.
 * @tparam Iter Number of loop iterations per warp (= Kp / 32).
 * @param[out] processorCounts Per-warp selected counts (device).
 * @param[in]  inputData       Input array to count (device, dataSize elements).
 * @param[in]  dataSize        Total number of input elements.
 */
template <typename T, int Iter>
__global__ void countKernel(unsigned int* processorCounts, const T* inputData, unsigned int dataSize) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto warpIdx = idx >> 5;
    const auto laneIdx = idx & 31;
    const auto begin = warpIdx * (Iter << 5);
    unsigned int count = 0;
#pragma unroll
    for (auto i = 0; i < Iter; ++i) {
        if (const auto current = begin + (i << 5) + laneIdx; current < dataSize) {
            if (inputData[current] > 0) {
                ++count;
            }
        }
    }

    count += __shfl_xor_sync(0xFFFFFFFF, count, 16);
    count += __shfl_xor_sync(0xFFFFFFFF, count, 8);
    count += __shfl_xor_sync(0xFFFFFFFF, count, 4);
    count += __shfl_xor_sync(0xFFFFFFFF, count, 2);
    count += __shfl_xor_sync(0xFFFFFFFF, count, 1);

    if (laneIdx == 0) {
        processorCounts[warpIdx] = count;
    }
}

/*!
 * @brief Launch countKernel with the grid dimensions derived from Kp and BLOCK_SIZE.
 *
 * @tparam T          Element type.
 * @tparam Kp         Elements processed per warp (must be a multiple of 32).
 * @tparam BLOCK_SIZE Threads per block.
 * @param[out] dProcessorCounts Per-warp selected counts (device, ceil(dataSize/Kp) entries).
 * @param[in]  dInputData       Input array (device).
 * @param[in]  dataSize         Total number of input elements.
 */
template <typename T, int Kp, int BLOCK_SIZE>
void launchCountKernel(unsigned int* dProcessorCounts, const T* dInputData, const unsigned int dataSize) {
    constexpr auto warpPerBlock = BLOCK_SIZE >> 5;
    const auto processors = (dataSize + Kp - 1) / Kp;
    const auto gridSize = (processors + warpPerBlock - 1) / warpPerBlock;
    constexpr auto Iter = Kp >> 5;
    countKernel<T, Iter><<<gridSize, BLOCK_SIZE>>>(dProcessorCounts, dInputData, dataSize);
}

/*!
 * @brief Block-level inclusive prefix-sum over per-warp counts (first pass).
 *
 * @details Computes an intra-warp scan, then an intra-block scan using shared
 * memory.  Writes per-block totals to buffer for the final-scan pass.
 * Must be followed by prefixSumFinalScanKernel to produce the globally
 * correct inclusive prefix-sum.
 *
 * @tparam T            Counter type (typically unsigned int).
 * @param[in,out] processorsCounts Per-warp counts; overwritten with block-local
 *                                  inclusive prefix-sum (device).
 * @param[out]    buffer            Per-block totals used by the final scan pass (device).
 * @param[in]     processors        Total number of warps.
 */
template <typename T>
__global__ void prefixSumKernel(T* processorsCounts, T* buffer, const unsigned int processors) {
    extern __shared__ T sharedBuffer[];
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto laneIdx = idx & 31;
    T localData = 0;
    if (idx < processors) {
        localData = processorsCounts[idx];
    }
    else {
        localData = 0;
    }

    // intra-warp scan and sum
    T temp;

    temp = __shfl_up_sync(0xFFFFFFFF, localData, 1);
    if (laneIdx >= 1) {
        localData += temp;
    }

    temp = __shfl_up_sync(0xFFFFFFFF, localData, 2);
    if (laneIdx >= 2) {
        localData += temp;
    }

    temp = __shfl_up_sync(0xFFFFFFFF, localData, 4);
    if (laneIdx >= 4) {
        localData += temp;
    }

    temp = __shfl_up_sync(0xFFFFFFFF, localData, 8);
    if (laneIdx >= 8) {
        localData += temp;
    }

    temp = __shfl_up_sync(0xFFFFFFFF, localData, 16);
    if (laneIdx >= 16) {
        localData += temp;
    }

    const auto warpIdxBlock = threadIdx.x >> 5;
    const auto numWarpsBlock = blockDim.x >> 5;
    // write to sharedBuffer
    if (laneIdx == 31) {
        sharedBuffer[warpIdxBlock] = localData;
    }

    __syncthreads();

    if (warpIdxBlock == 0) {
        T tempLocalData = (laneIdx < numWarpsBlock) ? sharedBuffer[laneIdx] : 0;
        // intra-warp scan
        T tempTemp;
        tempTemp = __shfl_up_sync(0xFFFFFFFF, tempLocalData, 1);
        if (laneIdx >= 1) {
            tempLocalData += tempTemp;
        }

        tempTemp = __shfl_up_sync(0xFFFFFFFF, tempLocalData, 2);
        if (laneIdx >= 2) {
            tempLocalData += tempTemp;
        }

        tempTemp = __shfl_up_sync(0xFFFFFFFF, tempLocalData, 4);
        if (laneIdx >= 4) {
            tempLocalData += tempTemp;
        }

        tempTemp = __shfl_up_sync(0xFFFFFFFF, tempLocalData, 8);
        if (laneIdx >= 8) {
            tempLocalData += tempTemp;
        }

        tempTemp = __shfl_up_sync(0xFFFFFFFF, tempLocalData, 16);
        if (laneIdx >= 16) {
            tempLocalData += tempTemp;
        }
        // write back
        if (laneIdx < numWarpsBlock) {
            sharedBuffer[laneIdx] = tempLocalData;
        }
        if (laneIdx == 31) {
            buffer[blockIdx.x] = tempLocalData;
        }
    }

    __syncthreads();

    // block-level scan
    if (warpIdxBlock > 0) {
        localData += sharedBuffer[warpIdxBlock - 1];
    }

    // write back
    if (idx < processors) {
        processorsCounts[idx] = localData;
    }
}

/*!
 * @brief Final-scan pass that adds inter-block offsets to produce a globally
 *        correct inclusive prefix-sum.
 *
 * @details Reads per-block totals from buffer (produced by prefixSumKernel),
 * accumulates the offset for each block, and adds it to the corresponding
 * entries in processorsCounts.
 *
 * @tparam T            Counter type.
 * @param[in,out] processorsCounts Block-local prefix-sum; updated to global (device).
 * @param[in]     buffer            Per-block totals from prefixSumKernel (device).
 * @param[in]     processors        Total number of warps.
 */
template <typename T>
__global__ void prefixSumFinalScanKernel(T* processorsCounts, const T* buffer, const unsigned int processors) {
    __shared__ T sharedBuffer;
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    const auto warpIdx = threadIdx.x >> 5;
    const auto laneIdx = threadIdx.x & 31;

    if (warpIdx == 0) {
        T addValue = 0;
        for (auto i = laneIdx; i < blockIdx.x; i += 32) {
            addValue += buffer[i];
        }

        addValue += __shfl_xor_sync(0xFFFFFFFF, addValue, 16);
        addValue += __shfl_xor_sync(0xFFFFFFFF, addValue, 8);
        addValue += __shfl_xor_sync(0xFFFFFFFF, addValue, 4);
        addValue += __shfl_xor_sync(0xFFFFFFFF, addValue, 2);
        addValue += __shfl_xor_sync(0xFFFFFFFF, addValue, 1);

        if (laneIdx == 0) {
            sharedBuffer = addValue;
        }
    }

    __syncthreads();

    if (idx < processors) {
        if (blockIdx.x > 0) {
            processorsCounts[idx] += sharedBuffer;
        }
    }
}

/*!
 * @brief Run the two-pass prefix-sum (prefixSumKernel + prefixSumFinalScanKernel).
 *
 * @details Allocates a temporary dBuffer for inter-block totals, launches both
 * kernel passes with cudaDeviceSynchronize between them, then frees dBuffer.
 * After this call dProcessorCounts holds the globally correct inclusive
 * prefix-sum, ready for use as output offsets in a scatter kernel.
 *
 * @tparam T          Counter type.
 * @tparam BLOCK_SIZE Threads per block.
 * @param[in,out] dProcessorCounts Per-warp counts on entry; global inclusive
 *                                  prefix-sum on exit (device).
 * @param[in]     processors        Number of warps (entries in dProcessorCounts).
 */
template <typename T, int BLOCK_SIZE>
void launchPrefixSumKernel(T* dProcessorCounts, const unsigned int processors) {
    constexpr auto warpPerBlock = BLOCK_SIZE >> 5;
    const auto sharedMemSize = warpPerBlock * sizeof(T);
    const auto gridSize = (processors + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const auto bufferSize = gridSize;
    T* dBuffer;
    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize * sizeof(T)));
    prefixSumKernel<T><<<gridSize, BLOCK_SIZE, sharedMemSize>>>(dProcessorCounts, dBuffer, processors);
    CUDA_CHECK(cudaDeviceSynchronize());
    prefixSumFinalScanKernel<T><<<gridSize, BLOCK_SIZE>>>(dProcessorCounts, dBuffer, processors);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(dBuffer));
}

/*!
 * @brief Scatter positive elements to contiguous output positions.
 *
 * @details Each warp uses the prefix-sum offsets in processorCounts to scatter
 * elements with value > 0 into outputData without gaps.  The output is a
 * densely packed array of the selected elements in their original order.
 *
 * @tparam T    Element type; elements with value > 0 are scattered.
 * @tparam Iter Number of loop iterations per warp (= Kp / 32).
 * @param[out] outputData       Compacted output array (device).
 * @param[in]  inputData        Input array (device, dataSize elements).
 * @param[in]  processorCounts  Global inclusive prefix-sum of per-warp counts (device).
 * @param[in]  dataSize         Total number of input elements.
 */
template <typename T, int Iter>
__global__ void compactPrefixKernel(T* outputData, const T* inputData, const unsigned int* processorCounts,
                                    const unsigned int dataSize) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto warpIdx = idx >> 5;
    const auto laneIdx = idx & 31;
    const auto begin = warpIdx * (Iter << 5);
    auto outputBegin = warpIdx > 0 ? processorCounts[warpIdx - 1] : 0;

#pragma unroll
    for (auto i = 0; i < Iter; ++i) {
        T localData = 0;
        unsigned int localFlags = 0;
        if (const auto current = begin + (i << 5) + laneIdx; current < dataSize) {
            localData = inputData[current];
            if (localData > 0) {
                localFlags = 1;
            }
        }
        // prefix Sum
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
        if (localData > 0) {
            const auto outputIdx = outputBegin + localFlags - 1;
            outputData[outputIdx] = localData;
        }
        const auto numCurIter = __shfl_sync(0xFFFFFFFF, localFlags, 31);
        outputBegin += numCurIter;
    }
}

/*!
 * @brief Launch compactPrefixKernel with grid dimensions derived from Kp and BLOCK_SIZE.
 *
 * @tparam T          Element type.
 * @tparam Kp         Elements processed per warp.
 * @tparam BLOCK_SIZE Threads per block.
 * @param[out] dOutputData       Compacted output array (device).
 * @param[in]  dInputData        Input array (device).
 * @param[in]  dProcessorCounts  Global inclusive prefix-sum of per-warp counts (device).
 * @param[in]  dataSize          Total number of input elements.
 */
template <typename T, int Kp, int BLOCK_SIZE>
void launchCompactPrefixKernel(T* dOutputData, const T* dInputData, const unsigned int* dProcessorCounts,
                               const unsigned int dataSize) {
    constexpr auto warpPerBlock = BLOCK_SIZE >> 5;
    const auto processors = (dataSize + Kp - 1) / Kp;
    const auto gridSize = (processors + warpPerBlock - 1) / warpPerBlock;
    constexpr auto Iter = Kp >> 5;
    compactPrefixKernel<T, Iter><<<gridSize, BLOCK_SIZE>>>(dOutputData, dInputData, dProcessorCounts, dataSize);
}

/**
 * @struct CompactHandler
 * @brief Legacy stateful compaction handler; pre-allocates count and prefix-sum buffers.
 *
 * @details Allocates processorCounts and buffer once in setup(), then reuses
 * them across compact() calls.  Not used by the current query pipeline
 * (which calls launchCountKernel and launchPrefixSumKernel directly) but
 * retained for compatibility with older code paths.
 *
 * @tparam Kp         Elements processed per warp.
 * @tparam BLOCK_SIZE Threads per block.
 */
template <int Kp, int BLOCK_SIZE>
struct CompactHandler {

    CompactHandler() : size(0), processorCounts(nullptr), buffer(nullptr) {}

    ~CompactHandler() {
        if (processorCounts) {
            cudaFree(processorCounts);
        }
        if (buffer) {
            cudaFree(buffer);
        }
    }

    void setup(const unsigned int dataSize) {
        size = dataSize;
        const auto processors = (size + Kp - 1) / Kp;
        const auto bufferSize = (processors + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cudaMalloc(&buffer, bufferSize * sizeof(unsigned int));
        cudaMalloc(&processorCounts, processors * sizeof(unsigned int));
    }

    template <typename T>
    void compact(T* outputData, const T* inputData) {
        constexpr auto warpPerBlock = BLOCK_SIZE >> 5;
        const auto processors = (size + Kp - 1) / Kp;
        const auto gridSize1 = (processors + warpPerBlock - 1) / warpPerBlock;
        const auto gridSize2 = (processors + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const auto sharedMemSize = warpPerBlock * sizeof(unsigned int);
        const auto iter = Kp >> 5;
        T* dInputData;
        auto* hProcessorCounts = new unsigned int[processors];
        cudaMalloc(&dInputData, size * sizeof(T));
        cudaMemcpy(dInputData, inputData, size * sizeof(T), cudaMemcpyHostToDevice);
        countKernel<T, iter><<<gridSize1, BLOCK_SIZE>>>(processorCounts, dInputData, size);
        prefixSumKernel<unsigned int><<<gridSize2, BLOCK_SIZE, sharedMemSize>>>(processorCounts, buffer, processors);
        prefixSumFinalScanKernel<unsigned int><<<gridSize2, BLOCK_SIZE>>>(processorCounts, buffer, processors);
        cudaMemcpy(hProcessorCounts, processorCounts, processors * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        const auto outputSize = hProcessorCounts[processors - 1];
        delete[] hProcessorCounts;
        T* dOutputData;
        cudaMalloc(&dOutputData, outputSize * sizeof(T));
        outputData = new T[outputSize];
        compactPrefixKernel<T, iter><<<gridSize1, BLOCK_SIZE>>>(dOutputData, dInputData, processorCounts, size);
        cudaMemcpy(outputData, dOutputData, outputSize * sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(dOutputData);
        cudaFree(dInputData);
    }

    unsigned int size;
    unsigned int* processorCounts;
    unsigned int* buffer;
};
#endif // CUDADB_COMPACT_CUH
