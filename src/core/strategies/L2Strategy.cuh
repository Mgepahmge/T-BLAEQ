/**
 * @file L2Strategy.cuh
 * @brief L2 memory strategy: selective lazy-load of per-level data.
 *
 * @details L2Strategy permanently caches only the small, level-independent
 * data (maps, coarsestMesh) on the device.  Everything else is loaded on
 * demand and freed immediately after use, minimising peak device memory.
 */

#pragma once

#include "IQueryStrategy.cuh"
#include "src/core/IndexData.cuh"

/**
 * @class L2Strategy
 * @brief Selective lazy-load strategy: only maps and coarsestMesh are permanent.
 *
 * @details L2Strategy sits between L1 (full permanent cache) and L3 (full
 * tiling) in the memory-performance trade-off:
 *
 * - Maps and coarsestMesh are uploaded once during prepare() and kept
 *   permanently, which is always safe since they are small.
 *
 * - Radius is uploaded transiently before pruning and freed immediately
 *   after the pruning kernel completes, before compact begins.
 *
 * - P-tensor values are never uploaded wholesale.  SpTSpMMultiplication_v3_L2
 *   downloads the pruned centroid ids, collects only the corresponding P-tensor
 *   columns from host memory, and uploads that compact subset.  This avoids
 *   the cost of transferring the full P-tensor array when pruning is effective
 *   (i.e. selectCount << col_nums), as is typical for KNN queries.
 *
 * - currentGrid is always device-resident because L2 is never assigned to a
 *   level whose predecessor could be L3 (the scheduler assigns policies
 *   coarsest-to-finest, so L2 is only placed above L2 or better levels).
 *
 * LevelResult::ownsIds is true because rowInd is dynamically allocated inside
 * SpTSpMMultiplication_v3_L2.
 */
class L2Strategy final : public IQueryStrategy {
public:
    explicit L2Strategy(IndexData& idx) : idx_(idx) {}

    /*!
     * @brief Upload permanently-resident data (maps, coarsestMesh) to the device.
     *
     * @details P vals and radius are intentionally not uploaded here; they are
     * managed transiently in runLevel().
     *
     * @param[in] reportOs Stream to write the memory policy report to.
     */
    void prepare(std::ostream& reportOs) override;

    /*!
     * @brief Execute one hierarchy level using lazy-loaded per-level data.
     *
     * @details Execution order:
     *   1. Upload radius -> run pruning kernel -> free radius immediately.
     *   2. Compact (currentGrid is device-resident).
     *   3. SpTSpMMultiplication_v3_L2: collect and upload only the P-tensor
     *      columns for pruned centroids -> run SpTSpM kernel -> free on return.
     *   4. Refactor using permanent dMaps.
     *
     * @param[in]  l           Level index (0 = coarsest).
     * @param[in]  currentGrid Device-resident input grid from the previous level.
     * @param[in]  qType       Query type (RANGE or KNN).
     * @param[in]  lo          Lower bounds of the query box (RANGE only).
     * @param[in]  hi          Upper bounds of the query box (RANGE only).
     * @param[in]  queryPoint  Query point coordinates (KNN only).
     * @param[in]  K           Number of nearest neighbours (KNN only).
     * @return LevelResult with ownsIds = true (ids_ allocated by SpTSpM_v3_L2).
     */
    LevelResult runLevel(size_t l, SparseGrid* currentGrid, QueryType qType, const double* lo, const double* hi,
                         const double* queryPoint, size_t K) override;

private:
    IndexData& idx_; //!< Reference to the shared index data container.
};
