/**
 * @file L3Strategy.cuh
 * @brief L3 memory strategy: unconditional tiling fallback for constrained devices.
 *
 * @details L3Strategy is the safety net of the memory hierarchy. It uploads
 * nothing during prepare() and manages all data in tiles sized dynamically
 * to the available device memory at the time of each operation.
 */

#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <iosfwd>
#include "IQueryStrategy.cuh"
#include "src/core/IndexData.cuh"

/**
 * @class L3Strategy
 * @brief Tiling fallback strategy: nothing is permanently resident on the device.
 *
 * @details L3Strategy is selected when even a single level's transient peak
 * memory (P vals + SpTSpM working arrays) does not fit in device memory.
 * Every kernel invocation is preceded by a cudaMemGetInfo call to determine
 * the current available budget, and data is uploaded in tiles that fit within
 * that budget and freed immediately after each tile.
 *
 * Because L3 does not call uploadPermanentData(), the coarsestView grid
 * constructed by QueryEngine has nullptr ids_/vals_ for level 0.  L3
 * detects this and falls back to the host-resident coarsestMesh directly.
 *
 * For subsequent levels, L3 checks whether the incoming currentGrid is
 * device-resident (produced by an L0/L1/L2 predecessor level) and downloads
 * it to host before processing.
 *
 * The output grid is always host-resident (LevelResult::onHost == true),
 * with ids_ and vals_ allocated with new[].  LevelResult::ownsIds = true
 * so the caller frees them with delete[].
 *
 * The four tiled sub-operations are:
 *   - runRangePruningTiled / runKnnPruningTiled: upload centroid tiles with
 *     pre-gathered radius, run the pruning kernel, download the mask.
 *   - runCompactTiled: tile the input centroids through the compact kernel
 *     and append selected entries to a growing host output array.
 *   - runSpTSpMTiled: iterate P columns in tiles; each tile uploads P vals,
 *     index arrays, and centroid vals, then downloads the output.  For KNN,
 *     distance computation and block-sort are fused on device before download.
 *   - runRefactorTiled: remaps output ids in-place on the host using the
 *     host-resident maps array (no device involvement needed).
 */
class L3Strategy final : public IQueryStrategy {
public:
    explicit L3Strategy(IndexData& idx) : idx_(idx) {}
    ~L3Strategy() override = default;

    /*!
     * @brief Compute statistics and print the policy report. No data is uploaded.
     *
     * @param[in] reportOs Stream to write the memory policy report to.
     */
    void prepare(std::ostream& reportOs) override;

    /*!
     * @brief Execute one hierarchy level using fully-tiled data management.
     *
     * @details Downloads currentGrid to host if device-resident, then runs
     * tiled pruning, compact, SpTSpM, and refactor.  The tile size for each
     * operation is determined independently by querying cudaMemGetInfo.
     *
     * @param[in]  l           Level index (0 = coarsest).
     * @param[in]  currentGrid Input grid; may be device- or host-resident.
     * @param[in]  qType       Query type (RANGE or KNN).
     * @param[in]  lo          Lower bounds of the query box (RANGE only).
     * @param[in]  hi          Upper bounds of the query box (RANGE only).
     * @param[in]  queryPoint  Query point coordinates (KNN only).
     * @param[in]  K           Number of nearest neighbours (KNN only).
     * @return LevelResult with onHost = true and ownsIds = true.
     */
    LevelResult runLevel(size_t l, SparseGrid* currentGrid, QueryType qType, const double* lo, const double* hi,
                         const double* queryPoint, size_t K) override;

private:
    IndexData& idx_; //!< Reference to the shared index data container.

    /*!
     * @brief Dispatch to runRangePruningTiled or runKnnPruningTiled.
     *
     * @param[in]  l               Level index.
     * @param[in]  hVals           Host centroid values (p * D doubles).
     * @param[in]  hIds            Host centroid global ids (p size_t values).
     * @param[in]  p               Number of centroids.
     * @param[in]  qType           Query type.
     * @param[in]  lo              Lower query bounds (RANGE only).
     * @param[in]  hi              Upper query bounds (RANGE only).
     * @param[in]  queryPoint      Query point (KNN only).
     * @param[in]  K               Number of nearest neighbours (KNN only).
     * @param[out] outSelectCount  Number of centroids passing the filter.
     * @return Host bool mask of length p; caller owns (delete[]).
     */
    bool* runPruningTiled(size_t l, const double* hVals, const size_t* hIds, size_t p, QueryType qType,
                          const double* lo, const double* hi, const double* queryPoint, size_t K,
                          size_t& outSelectCount);

    /*!
     * @brief Tiled RANGE pruning: uploads centroid tiles with pre-gathered radius.
     *
     * @details Before tiling, radius is gathered into a compact host array
     * (hRadiusGathered[i] = hRadius[hIds[i]]) to avoid indirect device
     * addressing.  lowBounds and upBounds are uploaded once before the loop.
     *
     * @param[in]  l               Level index.
     * @param[in]  hVals           Host centroid values.
     * @param[in]  hIds            Host centroid global ids.
     * @param[in]  p               Number of centroids.
     * @param[in]  lo              Lower query bounds.
     * @param[in]  hi              Upper query bounds.
     * @param[out] outSelectCount  Number of centroids passing the filter.
     * @return Host bool mask of length p; caller owns (delete[]).
     */
    bool* runRangePruningTiled(size_t l, const double* hVals, const size_t* hIds, size_t p, const double* lo,
                               const double* hi, size_t& outSelectCount);

    /*!
     * @brief Tiled KNN pruning: distance kernel + device sort + host STEP.
     *
     * @details Radius and clusterSizes are pre-gathered by global centroid id
     * to eliminate indirect device addressing.  Each tile computes distances
     * on device, block-sorts them with blockMergeSortKernel, then downloads
     * the sorted ClusterL3 array.  After all tiles are processed,
     * hostSerialMergeSort merges the tile-sorted segments and STEP selects
     * the final centroid set.
     *
     * @param[in]  l               Level index.
     * @param[in]  hVals           Host centroid values.
     * @param[in]  hIds            Host centroid global ids.
     * @param[in]  p               Number of centroids.
     * @param[in]  queryPoint      Query point.
     * @param[in]  K               Number of nearest neighbours.
     * @param[out] outSelectCount  Number of centroids selected by STEP.
     * @return Host bool mask of length p; caller owns (delete[]).
     */
    bool* runKnnPruningTiled(size_t l, const double* hVals, const size_t* hIds, size_t p, const double* queryPoint,
                             size_t K, size_t& outSelectCount);

    /*!
     * @brief Tiled compact: appends selected centroids to a growing host array.
     *
     * @details Each tile uploads a slice of hVals/hIds and the corresponding
     * mask slice to device, runs count -> prefix-sum -> scatter, downloads the
     * compacted output, and appends it at outVals[offset + written].
     *
     * @param[in]  hVals        Host input centroid values.
     * @param[in]  hIds         Host input centroid ids.
     * @param[in]  p            Total number of input centroids.
     * @param[in]  numRows      Number of rows in the grid (passed to compact kernel).
     * @param[in]  mask         Host bool selection mask of length p.
     * @param[in]  selectCount  Number of true entries in mask.
     * @param[out] outVals      Host output buffer for selected centroid values.
     * @param[out] outIds       Host output buffer for selected centroid ids.
     * @param[in]  offset       Write offset into outVals/outIds.
     * @return Number of entries appended.
     */
    size_t runCompactTiled(const double* hVals, const size_t* hIds, size_t p, size_t numRows, const bool* mask,
                           size_t selectCount, double* outVals, size_t* outIds, size_t offset);

    /*!
     * @brief Tiled SpTSpM: column-tiled matrix-vector multiply on device.
     *
     * @details Sorts pruned centroids by column index, then iterates over
     * groups of columns whose total fine-point count fits within the current
     * device memory budget.  Each tile uploads P vals and index arrays,
     * launches SpMSpVKernelAOS_v2, and downloads the output into host arrays.
     *
     * @param[in]  l           Level index.
     * @param[in]  prunedVals  Host pruned centroid values (pruneCount * D).
     * @param[in]  prunedIds   Host pruned centroid global ids.
     * @param[in]  pruneCount  Number of pruned centroids.
     * @param[out] outYVals    Allocated host output values (new[]); caller owns.
     * @param[out] outYIds     Allocated host output ids (new[]); caller owns.
     * @param[out] outTotalNnz Total number of output fine points.
     */
    void runSpTSpMTiled(size_t l, const double* prunedVals, const size_t* prunedIds, size_t pruneCount,
                        double*& outYVals, size_t*& outYIds, size_t& outTotalNnz);

    /*!
     * @brief Remap output ids in-place using the host-resident maps[l+1] array.
     *
     * @details Since maps[l+1] is always available as a host array in IndexData,
     * this operation is a pure host loop with no device involvement.
     *
     * @param[in]     l         Level index; maps[l+1] is used for remapping.
     * @param[in,out] yIds      Output fine-point ids to remap in-place.
     * @param[in]     totalNnz  Number of entries in yIds.
     */
    void runRefactorTiled(size_t l, size_t* yIds, size_t totalNnz);
};
