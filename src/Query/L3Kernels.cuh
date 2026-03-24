/**
 * @file L3Kernels.cuh
 * @brief CUDA kernels used exclusively by the L3 tiling strategy.
 *
 * @details These kernels are variants of the standard pruning kernels that
 * eliminate indirect addressing via an indexs[] array.  In the standard
 * kernels, radius is indexed as radius[indexs[i]], requiring the full global
 * radius array on the device.  In the L3 tiled path, radius is pre-gathered
 * on the host (hRadiusGathered[i] = hRadius[hIds[i]]) before each tile is
 * uploaded, so the kernel can use radius[i] directly without any indirection.
 * This removes the need to keep the global radius array on the device.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include "src/Query/check.cuh"

/*!
 * @brief Per-centroid range pruning kernel for the L3 tiled path.
 *
 * @details Identical pruning logic to rangePruningKernel, but radius[i] is
 * the pre-gathered max-radius of tile-local centroid i.  No indexs array
 * is needed.  All pointer arguments must reside in device memory.
 *
 * @param[out] mask       Output selection mask (p bools, device).
 * @param[in]  lowBounds  Lower bounds of the query box (dim doubles, device).
 * @param[in]  upBounds   Upper bounds of the query box (dim doubles, device).
 * @param[in]  centroids  Centroid coordinates in AOS layout (p * dim doubles, device).
 * @param[in]  radius     Pre-gathered max-radius array (p doubles, device).
 * @param[in]  dim        Data dimensionality.
 * @param[in]  p          Number of centroids in this tile.
 */
__global__ void rangePruningKernelL3(bool* mask, const double* lowBounds, const double* upBounds,
                                     const double* centroids, const double* radius, const size_t dim, const size_t p);

/**
 * @struct ClusterL3
 * @brief Distance-label pair for L3 KNN tiled pruning.
 *
 * @details Mirrors the Cluster struct from KNN.cuh but stores the global
 * centroid index (rStart + tile-local idx) in label rather than the
 * tile-local index.  This allows clusters from different tiles to be merged
 * and sorted together on the host after all tiles are processed.
 */
struct ClusterL3 {
    double distance; //!< Lower-bound distance from query to this centroid.
    uint64_t label; //!< Global centroid index: rStart + tile-local index.

    __device__ __host__ bool operator<(const ClusterL3& o) const { return distance < o.distance; }
    __device__ __host__ bool operator>(const ClusterL3& o) const { return distance > o.distance; }
    __device__ __host__ bool operator==(const ClusterL3& o) const { return distance == o.distance; }
    __device__ __host__ bool operator<=(const ClusterL3& o) const { return distance <= o.distance; }
    __device__ __host__ bool operator>=(const ClusterL3& o) const { return distance >= o.distance; }
};

/*!
 * @brief Per-centroid distance computation kernel for the L3 tiled KNN path.
 *
 * @details Same distance computation as calculateClusterDistanceKernel, but
 * radius[i] is pre-gathered (no indexs[] indirection) and the stored label
 * is rStart + i (global index) rather than i (tile-local index).  This
 * enables correct STEP selection after clusters from all tiles are merged.
 *
 * @param[out] clusters_  Output ClusterL3 array (p entries, device).
 * @param[in]  queryPoint Query coordinates (dim doubles, device).
 * @param[in]  centroids  Centroid coordinates in AOS layout (p * dim doubles, device).
 * @param[in]  radius     Pre-gathered max-radius array (p doubles, device).
 * @param[in]  dim        Data dimensionality.
 * @param[in]  p          Number of centroids in this tile.
 * @param[in]  rStart     Global index of the first centroid in this tile.
 */
__global__ void calculateClusterDistanceKernelL3(void* clusters_, const double* queryPoint, const double* centroids,
                                                 const double* radius, const size_t dim, const size_t p,
                                                 const size_t rStart);
