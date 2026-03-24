/**
 * @file StrategyCommon.cuh
 * @brief Shared utility functions used by L0, L1, and L2 strategy classes.
 *
 * @details This file declares free functions that encapsulate operations
 * common to the strategies that keep currentGrid on the device (L0, L1, L2).
 * They have no hidden state and operate purely on their arguments.
 * L3Strategy does not use these functions; it manages all operations through
 * its own tiled wrappers.
 */

#pragma once

#include <cstddef>
#include "src/Data_Structures/Data_Structures.cuh"
#include "src/Data_Structures/File.cuh"
#include "src/core/IndexData.cuh"

/*!
 * @brief Free a device-resident SparseGrid, respecting ownership flags.
 *
 * @details The two ownership flags exist because L0 returns grids whose
 * ids_ and vals_ point into IndexData-owned buffers (dSpTSpMBufs[l].rowInd
 * and l0Bufs.dYValue[l] respectively).  Calling cudaFree on those pointers
 * would corrupt the pre-allocated working storage.  Setting ownsIds = false
 * and ownsVals = false prevents the free while still deleting the SparseGrid
 * wrapper object itself.
 *
 * @param[in] g        The SparseGrid to free. No-op when nullptr.
 * @param[in] ownsIds  When true, ids_ is freed with cudaFree.
 *                     When false, ids_ is owned by IndexData and must not be freed.
 * @param[in] ownsVals When true, vals_ is freed with cudaFree.
 *                     When false, vals_ is owned by IndexData and must not be freed.
 */
void freeDeviceSparseGrid(SparseGrid* g, bool ownsIds, bool ownsVals = true);

/*!
 * @brief Run RANGE or KNN pruning on currentGrid and return a host selection mask.
 *
 * @details Dispatches to rangePruning or knnPruning based on qType.  All device
 * data (centroids, radius) comes from the already-uploaded IndexData buffers.
 * For KNN, hRadius and hClusterSizes are read directly from host memory,
 * avoiding any device-to-host transfer of those arrays.
 *
 * @param[in]  l             Level index (0 = coarsest).
 * @param[in]  idx           IndexData with device-resident radius for level l.
 * @param[in]  currentGrid   Device-resident input grid to prune.
 * @param[in]  qType         Query type (RANGE or KNN).
 * @param[in]  lo            Lower bounds of the query box (RANGE only).
 * @param[in]  hi            Upper bounds of the query box (RANGE only).
 * @param[in]  queryPoint    Query point coordinates (KNN only).
 * @param[in]  K             Number of nearest neighbours (KNN only).
 * @param[out] selectCount   Number of centroids passing the filter.
 * @return Host bool array of length currentGrid->nnz; caller owns (delete[]).
 */
bool* runPruning(size_t l, IndexData& idx, SparseGrid* currentGrid, QueryType qType, const double* lo, const double* hi,
                 const double* queryPoint, size_t K, size_t& selectCount);
