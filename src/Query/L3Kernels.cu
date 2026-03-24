#include "L3Kernels.cuh"

__global__ void rangePruningKernelL3(bool* mask, const double* lowBounds, const double* upBounds,
                                     const double* centroids, const double* radius, const size_t dim, const size_t p) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= p) {
        return;
    }

    double dist = 0.0;
    for (size_t d = 0; d < dim; ++d) {
        const double cv = centroids[idx * dim + d];
        const double lo = lowBounds[d];
        const double hi = upBounds[d];
        double diff = 0.0;
        if (cv < lo) {
            diff = lo - cv;
        }
        else if (cv > hi) {
            diff = cv - hi;
        }
        dist += diff * diff;
    }
    const double r = radius[idx];
    mask[idx] = (dist <= r * r);
}

__global__ void calculateClusterDistanceKernelL3(void* clusters_, const double* queryPoint, const double* centroids,
                                                 const double* radius, const size_t dim, const size_t p,
                                                 const size_t rStart) {
    auto* clusters = static_cast<ClusterL3*>(clusters_);
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= p) {
        return;
    }

    // Global label so that after merging tiles the original centroid is known
    clusters[idx].label = static_cast<uint64_t>(rStart + idx);

    double dist = 0.0;
    for (size_t d = 0; d < dim; ++d) {
        const double diff = queryPoint[d] - centroids[idx * dim + d];
        dist += diff * diff;
    }
    dist = __dsqrt_rd(dist);
    clusters[idx].distance = dist - radius[idx]; // radius is pre-gathered
}
