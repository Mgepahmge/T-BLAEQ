/**
 * @file KNN.cuh
 * @brief KNN pruning kernel, Cluster type, and STEP-based wrapper.
 *
 * @details Provides the distance computation kernel, the Cluster type used
 * for sorting, and the knnPruning wrapper that implements the full
 * distance-sort-STEP pipeline for K-nearest-neighbour queries.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include "src/Query/check.cuh"

/**
 * @struct Cluster
 * @brief Distance-label pair used for KNN centroid sorting and STEP selection.
 *
 * @details Stores the lower-bound distance from a query point to a cluster
 * (centroid distance minus max-radius) and the global centroid label.
 * Comparison operators enable sorting by distance on both host and device.
 */
struct Cluster {
    double distance; //!< Lower-bound distance: dist(query, centroid) - radius.
    uint64_t label; //!< Global centroid index in the current grid.

    __device__ __host__ bool operator<(const Cluster& o) const { return distance < o.distance; }
    __device__ __host__ bool operator>(const Cluster& o) const { return distance > o.distance; }
    __device__ __host__ bool operator==(const Cluster& o) const { return distance == o.distance; }
    __device__ __host__ bool operator<=(const Cluster& o) const { return distance <= o.distance; }
    __device__ __host__ bool operator>=(const Cluster& o) const { return distance >= o.distance; }
};

/*!
 * @brief Compute lower-bound distances from the query point to each centroid.
 *
 * @details For each centroid i, computes sqrt(sum_d (query[d] - centroid[i*D+d])^2)
 * minus radius[indexs[i]], and stores the result in clusters[i].distance with
 * clusters[i].label = i.  All pointer arguments must reside in device memory.
 *
 * @param[out] clusters_   Output array of Cluster structs (p entries, device).
 * @param[in]  queryPoint  Query coordinates (dim doubles, device).
 * @param[in]  centroids   Centroid coordinates in AOS layout (p * dim doubles, device).
 * @param[in]  dRadius     Max-radius array (length doubles, device), indexed via indexs.
 * @param[in]  dim         Data dimensionality.
 * @param[in]  indexs      Global centroid-to-radius index map (p size_t, device).
 * @param[in]  p           Number of centroids.
 * @note This kernel must not be modified.
 */
__global__ void calculateClusterDistanceKernel(void* clusters_, const double* queryPoint, const double* centroids,
                                               const double* dRadius, const size_t dim, const size_t* indexs,
                                               const size_t p);

/*!
 * @brief Run KNN pruning using the STEP algorithm and return a host selection mask.
 *
 * @details Executes four steps:
 *   1. Distance kernel: computes lower-bound distances for all p centroids.
 *   2. Device sort: sorts the Cluster array by distance using gpuSort.
 *   3. STEP selection: iterates sorted clusters on host, accumulating fine-point
 *      counts until K neighbours are guaranteed, then stops.
 *   4. Mask construction: marks selected cluster labels true in the output mask.
 *
 * The radius array is used in two places with different residency requirements.
 * To avoid a full device-to-host transfer of the radius array, the caller
 * supplies both a device copy (dRadius, for the kernel) and a host copy
 * (hRadius, for STEP).  hClusterSizes is used only by STEP and is never
 * uploaded to the device.
 *
 * Used by L1Strategy and L2Strategy via StrategyCommon::runPruning().
 * L3Strategy uses its own tiled variant (runKnnPruningTiled) instead.
 *
 * @param[in]  k                Number of nearest neighbours requested.
 * @param[in]  p                Number of centroids to evaluate.
 * @param[in]  dim              Data dimensionality.
 * @param[in]  length           Total number of centroids (length of radius arrays).
 * @param[in]  queryPoint       Query coordinates (dim doubles, host).
 * @param[in]  centroids        Centroid coordinates (p * dim doubles, device).
 * @param[in]  dRadius          Max-radius array (length doubles, device).
 * @param[in]  hRadius          Same data as dRadius on host; used by STEP.
 * @param[in]  hClusterSizes    Fine-point count per centroid (length size_t, host).
 * @param[in]  indexs           Global centroid ids of the p centroids (device).
 * @param[out] outSelectedCount Number of centroids retained after STEP.
 * @return Host bool array of length p; caller owns (delete[]).
 */
bool* knnPruning(size_t k, size_t p, size_t dim, size_t length, const double* queryPoint, const double* centroids,
                 const double* dRadius, const double* hRadius, const size_t* hClusterSizes, const size_t* indexs,
                 size_t& outSelectedCount);
