#include <algorithm>
#include <limits>
#include <vector>
#include "KNN.cuh"
#include "check.cuh"
#include "src/MergeSort/MergeSort.cuh"
#include "src/utils/NVTXProfiler.cuh"

__global__ void calculateClusterDistanceKernel(void* clusters_, const double* queryPoint, const double* centroids,
                                               const double* dRadius, const size_t dim, const size_t* indexs,
                                               const size_t p) {
    auto* clusters = static_cast<Cluster*>(clusters_);
    if (const auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < p) {
        const auto index = indexs[idx];
        clusters[idx].label = idx;
        double dist = 0.0;
        for (size_t d = 0; d < dim; ++d) {
            const auto diff = queryPoint[d] - centroids[idx * dim + d];
            dist += diff * diff;
        }
        dist = __dsqrt_rd(dist);
        clusters[idx].distance = dist - dRadius[index];
    }
}

bool* knnPruning(size_t k, size_t p, size_t dim, size_t length, const double* queryPoint, const double* centroids,
                 const double* dRadius, const double* hRadius, const size_t* hClusterSizes, const size_t* indexs,
                 size_t& outSelectedCount) {
    NvtxProfiler profiler("knnPruning", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Blue);

    // Calculate distance
    Cluster* dClusters = nullptr;
    CUDA_CHECK(cudaMalloc(&dClusters, p * sizeof(Cluster)));

    {
        NvtxProfiler p1("knn_distance", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::SkyBlue);

        double* dQueryPoint = nullptr;
        CUDA_CHECK(cudaMalloc(&dQueryPoint, dim * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(dQueryPoint, queryPoint, dim * sizeof(double), cudaMemcpyHostToDevice));

        const dim3 block(256);
        const dim3 grid((p + 255) / 256);
        calculateClusterDistanceKernel<<<grid, block>>>(dClusters, dQueryPoint, centroids, dRadius, dim, indexs, p);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(dQueryPoint));
    }

    // Sort clusters by distance on device
    {
        NvtxProfiler p2("knn_sort", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::SpringGreen);

        constexpr int kIPB = 512;
        constexpr int kIPT = 2;
        Cluster sentinel{};
        sentinel.distance = std::numeric_limits<double>::max();
        sentinel.label = 0;
        gpuSort<kIPB, kIPT, true>(dClusters, dClusters, p, sentinel);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Download only what STEP needs from device
    auto* hClusters = new Cluster[p];
    auto* hIndexs = new size_t[p];

    CUDA_CHECK(cudaMemcpy(hClusters, dClusters, p * sizeof(Cluster), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hIndexs, indexs, p * sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dClusters));

    // STEP pruning on host
    {
        NvtxProfiler p3("knn_step", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::LimeGreen);
    }

    size_t currentCount = 0;
    double currentDistance = 0.0;
    std::vector<size_t> selectedLabels;
    size_t selectedCount = 0;

    for (size_t i = 0; i < p; ++i) {
        const size_t index = hIndexs[hClusters[i].label];
        if (currentCount >= k) {
            if (hClusters[i].distance > currentDistance) {
                break;
            }
            currentCount += hClusterSizes[index];
            selectedLabels.push_back(hClusters[i].label);
            ++selectedCount;
        }
        else {
            currentCount += hClusterSizes[index];
            currentDistance = std::max(currentDistance, hClusters[i].distance + 2.0 * hRadius[index]);
            selectedLabels.push_back(hClusters[i].label);
            ++selectedCount;
        }
    }

    // Build result mask
    delete[] hClusters;
    delete[] hIndexs;

    auto* result = new bool[p];
    std::fill(result, result + p, false);
    for (const auto label : selectedLabels) {
        result[label] = true;
    }

    outSelectedCount = selectedCount;
    return result;
}
