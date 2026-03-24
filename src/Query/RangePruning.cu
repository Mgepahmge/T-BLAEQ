#include <algorithm>
#include "RangePruning.cuh"
#include "src/utils/NVTXProfiler.cuh"

__global__ void rangePruningKernel(bool* mask, const double* lowBounds, const double* upBounds, const double* centroids,
                                   const double* radius, const size_t dim, const size_t p, const size_t* indexs) {
    if (const auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < p) {
        double dist = 0.0;
        for (auto d = 0; d < dim; ++d) {
            double diff = 0.0;
            const auto cv = centroids[idx * dim + d];
            const auto lo = lowBounds[d];
            const auto hi = upBounds[d];
            if (cv < lo) {
                diff = lo - cv;
            }
            else if (cv > hi) {
                diff = cv - hi;
            }
            dist += diff * diff;
        }
        const auto r = radius[indexs[idx]];
        mask[idx] = (dist <= r * r);
    }
}

bool* rangePruning(const double* lowBounds, const double* upBounds, size_t dim, const double* centroids,
                   const double* radius, size_t p, const size_t* indexs, size_t length, size_t& outSelectedCount) {
    NvtxProfiler profiler("rangePruning", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Blue);

    // Allocate device output mask
    bool* dMask = nullptr;
    CUDA_CHECK(cudaMalloc(&dMask, p * sizeof(bool)));

    // Launch kernel
    const dim3 block(256);
    const dim3 grid((p + 255) / 256);
    rangePruningKernel<<<grid, block>>>(dMask, lowBounds, upBounds, centroids, radius, dim, p, indexs);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result to host
    auto* mask = new bool[p];
    CUDA_CHECK(cudaMemcpy(mask, dMask, p * sizeof(bool), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dMask));

    outSelectedCount = static_cast<size_t>(std::count(mask, mask + p, true));
    return mask;
}
