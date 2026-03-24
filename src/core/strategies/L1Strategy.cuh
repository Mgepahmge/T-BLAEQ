/**
 * @file L1Strategy.cuh
 * @brief L1 memory strategy: all index data permanently cached on device.
 *
 * @details L1Strategy uploads all P-tensor values, radius arrays, maps, and
 * the coarsest mesh to the device during prepare() and keeps them resident
 * for the lifetime of the index.  Per-query allocation is limited to the
 * SpTSpM index arrays and output buffer inside SpTSpMMultiplication_v3.
 */

#pragma once

#include "IQueryStrategy.cuh"
#include "src/core/IndexData.cuh"

/**
 * @class L1Strategy
 * @brief Full-cache strategy: all index data permanently resident on the device.
 *
 * @details After prepare(), the device holds P vals, radius, maps, and
 * coarsestMesh for all levels.  runLevel() calls the standard wrappers
 * (rangePruning/knnPruning, compactGrid, SpTSpMMultiplication_v3, refactor)
 * without any additional upload/free overhead.  Dynamic allocation occurs
 * only inside SpTSpMMultiplication_v3 for the per-query SpTSpM index arrays
 * and the output yValue buffer.
 *
 * LevelResult::ownsIds is true because rowInd is dynamically allocated inside
 * SpTSpMMultiplication_v3 and transferred to the returned SparseGrid.
 */
class L1Strategy final : public IQueryStrategy {
public:
    explicit L1Strategy(IndexData& idx) : idx_(idx) {}

    /*!
     * @brief Upload all index data to the device as permanently-resident buffers.
     *
     * @details Calls uploadPermanentData() for maps and coarsestMesh, then
     * uploads P vals and radius for every level.
     *
     * @param[in] reportOs Stream to write the memory policy report to.
     */
    void prepare(std::ostream& reportOs) override;

    /*!
     * @brief Execute one hierarchy level using permanently-cached device data.
     *
     * @details Calls pruning, compact, SpTSpMMultiplication_v3, and refactor
     * in sequence.  All large data (P vals, radius, maps) is already on the
     * device; only the SpTSpM working arrays are allocated per call.
     *
     * @param[in]  l           Level index (0 = coarsest).
     * @param[in]  currentGrid Device-resident input grid from the previous level.
     * @param[in]  qType       Query type (RANGE or KNN).
     * @param[in]  lo          Lower bounds of the query box (RANGE only).
     * @param[in]  hi          Upper bounds of the query box (RANGE only).
     * @param[in]  queryPoint  Query point coordinates (KNN only).
     * @param[in]  K           Number of nearest neighbours (KNN only).
     * @return LevelResult with ownsIds = true (ids_ allocated by SpTSpM_v3).
     */
    LevelResult runLevel(size_t l, SparseGrid* currentGrid, QueryType qType, const double* lo, const double* hi,
                         const double* queryPoint, size_t K) override;

private:
    IndexData& idx_; //!< Reference to the shared index data container.
};
