/**
 * @file RangePruning.cuh
 * @brief Range-query pruning kernel and wrapper for the hierarchical index.
 *
 * @details Provides the CUDA kernel and host wrapper that eliminate clusters
 * whose maximum possible extent (centroid +/- radius sphere) does not
 * overlap the axis-aligned query box.  Each centroid is processed
 * independently, so the kernel is trivially parallelisable.
 */

#pragma once

#include <cstddef>
#include "src/Query/check.cuh"

/*!
 * @brief Per-centroid range pruning kernel.
 *
 * @details For each centroid i, computes the minimum squared distance from
 * the centroid to the query box and compares it against radius[indexs[i]]^2.
 * Sets mask[i] = true when the cluster may intersect the query box.
 * All pointer arguments must reside in device memory.
 *
 * @param[out] mask       Output selection mask (p bools).
 * @param[in]  lowBounds  Lower bounds of the query box (dim doubles).
 * @param[in]  upBounds   Upper bounds of the query box (dim doubles).
 * @param[in]  centroids  Centroid coordinates in AOS layout (p * dim doubles).
 * @param[in]  radius     Max-radius array (length doubles), indexed via indexs.
 * @param[in]  dim        Data dimensionality.
 * @param[in]  p          Number of centroids.
 * @param[in]  indexs     Global centroid-to-radius index map (p size_t values).
 * @note This kernel must not be modified.
 */
__global__ void rangePruningKernel(bool* mask, const double* lowBounds, const double* upBounds, const double* centroids,
                                   const double* radius, const size_t dim, const size_t p, const size_t* indexs);

/*!
 * @brief Run range pruning on a set of centroids and return a host selection mask.
 *
 * @details Launches rangePruningKernel, downloads the result mask to host,
 * and counts the number of selected centroids.  All device pointer arguments
 * must already reside in device memory before this function is called.
 *
 * Used by L1Strategy and L2Strategy via StrategyCommon::runPruning().
 * L3Strategy uses its own tiled variant (rangePruningKernelL3) instead.
 *
 * @param[in]  lowBounds        Lower bounds of the query box (dim doubles, device).
 * @param[in]  upBounds         Upper bounds of the query box (dim doubles, device).
 * @param[in]  dim              Data dimensionality.
 * @param[in]  centroids        Centroid coordinates (p * dim doubles, device).
 * @param[in]  radius           Max-radius array (length doubles, device).
 * @param[in]  p                Number of centroids.
 * @param[in]  indexs           Global centroid-to-radius index map (p size_t, device).
 * @param[in]  length           Number of entries in the radius array.
 * @param[out] outSelectedCount Number of true entries in the returned mask.
 * @return Host bool array of length p; caller owns (delete[]).
 */
bool* rangePruning(const double* lowBounds, const double* upBounds, size_t dim, const double* centroids,
                   const double* radius, size_t p, const size_t* indexs, size_t length, size_t& outSelectedCount);
