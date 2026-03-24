/**
 * @file L0Strategy.cuh
 * @brief L0 memory strategy: zero cudaMalloc in the query hot path.
 *
 * @details L0Strategy achieves zero dynamic allocation during query execution
 * by pre-allocating all working buffers in IndexData::l0Bufs during prepare().
 * These buffers are sized to the maximum required capacity across all levels
 * and are reused for every query without any reallocation.
 */

#pragma once

#include <iosfwd>
#include "IQueryStrategy.cuh"
#include "src/core/IndexData.cuh"

/**
 * @class L0Strategy
 * @brief Highest-performance memory strategy: all working buffers pre-allocated.
 *
 * @details During prepare(), L0Strategy uploads P vals and radius permanently
 * and calls allocSpTSpMBuffers() and allocL0WorkBuffers() to pre-size all
 * device and host working buffers.  runLevel() then performs:
 *   1. Pruning using pre-allocated dMask, dClusters, dQueryPoint, dLo, dHi.
 *   2. Compact using pre-allocated dProcCounts, dCompactIds, dCompactVals.
 *   3. SpTSpM via SpTSpMMultiplication_v3_L0_nb using pre-allocated
 *      dSpTSpMBufs[l] index arrays and l0Bufs.dYValue[l] for output.
 *   4. Refactor using permanently-resident dMaps.
 *
 * The returned fineGrid's ids_ points to dSpTSpMBufs[l].rowInd and vals_ points
 * to l0Bufs.dYValue[l], both owned by IndexData.  Therefore LevelResult is
 * returned with ownsIds = false and ownsVals = false.
 */
class L0Strategy final : public IQueryStrategy {
public:
    explicit L0Strategy(IndexData& idx) : idx_(idx) {}
    ~L0Strategy() override = default;

    /*!
     * @brief Upload all persistent data and pre-allocate all working buffers.
     *
     * @details Uploads permanent data (maps, coarsestMesh), then uploads P vals
     * and radius for all levels, allocates SpTSpM index buffers, and allocates
     * the L0 working storage in IndexData::l0Bufs.
     *
     * @param[in] reportOs Stream to write the memory policy report to.
     */
    void prepare(std::ostream& reportOs) override;

    /*!
     * @brief Execute one hierarchy level with zero device memory allocation.
     *
     * @details Uses only pre-allocated buffers from IndexData::l0Bufs and
     * IndexData::dSpTSpMBufs[l].  No cudaMalloc or cudaFree is called.
     *
     * @param[in]  l           Level index (0 = coarsest).
     * @param[in]  currentGrid Device-resident input grid from the previous level.
     * @param[in]  qType       Query type (RANGE or KNN).
     * @param[in]  lo          Lower bounds of the query box (RANGE only).
     * @param[in]  hi          Upper bounds of the query box (RANGE only).
     * @param[in]  queryPoint  Query point coordinates (KNN only).
     * @param[in]  K           Number of nearest neighbours (KNN only).
     * @return LevelResult with ownsIds = false and ownsVals = false, because
     *         both ids_ and vals_ of the returned grid are owned by IndexData.
     */
    LevelResult runLevel(size_t l, SparseGrid* currentGrid, QueryType qType, const double* lo, const double* hi,
                         const double* queryPoint, size_t K) override;

private:
    IndexData& idx_; //!< Reference to the shared index data container.
};
