#include "GridCompact.cuh"
#include "compact.cuh"
#include "src/Data_Structures/Data_Structures.cuh"
#include "src/utils/NVTXProfiler.cuh"

SparseGrid* compactGrid(const SparseGrid& grid, const bool* mask, size_t validCount) {
    NvtxProfiler profiler("compactGrid", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Yellow);

    const size_t len = grid.get_num_rows();
    const size_t dim = grid.get_dimensions();
    const size_t nnz = grid.get_nnz_nums();

    // ids and vals are already on device
    const size_t* dIndex = grid.get_ids_();
    const double* dData = grid.get_vals_();

    // Upload mask to device
    bool* dMask = nullptr;
    CUDA_CHECK(cudaMalloc(&dMask, nnz * sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(dMask, mask, nnz * sizeof(bool), cudaMemcpyHostToDevice));

    // Step 1: per-warp element counts
    constexpr int kBlockSize = 512;
    constexpr int kKp = 128;
    const size_t nProc = (nnz + kKp - 1) / kKp;

    unsigned int* procCounts = nullptr;
    CUDA_CHECK(cudaMalloc(&procCounts, nProc * sizeof(unsigned int)));
    launchCountKernel<bool, kKp, kBlockSize>(procCounts, dMask, nnz);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: prefix scan
    launchPrefixSumKernel<unsigned int, kBlockSize>(procCounts, nProc);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 3: scatter
    constexpr int kWarpPerBlock = kBlockSize >> 5;
    constexpr int kIter = kKp >> 5;
    const size_t gridSz = (nProc + kWarpPerBlock - 1) / kWarpPerBlock;

    size_t* dOutIndex = nullptr;
    double* dOutData = nullptr;
    CUDA_CHECK(cudaMalloc(&dOutIndex, validCount * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&dOutData, validCount * dim * sizeof(double)));

    gridCompactPrefixKernel<kIter><<<gridSz, kBlockSize>>>(dOutData, dOutIndex, dData, dIndex, dMask, procCounts,
                                                           static_cast<unsigned int>(nnz), dim, nProc, validCount);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(dMask));
    CUDA_CHECK(cudaFree(procCounts));

    // Wrap device pointers in a SparseGrid (ownsBuffers = false)
    return new SparseGrid{len, dim, validCount, dOutIndex, dOutData};
}
