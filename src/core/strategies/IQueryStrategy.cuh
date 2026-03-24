/**
 * @file IQueryStrategy.cuh
 * @brief Defines the strategy interface and result type for per-level query execution.
 *
 * @details This file contains the pure-virtual interface IQueryStrategy and the
 * LevelResult struct that carries the output of a single hierarchy level.
 * The strategy pattern decouples the memory management policy (L0/L1/L2/L3)
 * from the query execution logic in QueryEngine.  Each concrete strategy
 * decides how and when data is uploaded to and freed from device memory.
 */

#pragma once

#include <cstddef>
#include <iosfwd>
#include "src/Data_Structures/Data_Structures.cuh"
#include "src/Data_Structures/File.cuh"

/**
 * @struct LevelResult
 * @brief Output of a single runLevel() call: the fine-mesh SparseGrid together
 *        with ownership metadata required for safe cleanup.
 *
 * @details Three boolean flags describe the memory layout of the returned grid:
 *
 * ownsIds  - When false, ids_ points into a buffer owned by IndexData
 *            (e.g. L0's dSpTSpMBufs[l].rowInd).  The caller must not free it.
 *            When true, ids_ was allocated by the strategy and must be freed
 *            by the caller.
 *
 * ownsVals - Same semantics as ownsIds but for vals_.  False for L0, whose
 *            vals_ points into IndexData::l0Bufs.dYValue[l].
 *
 * onHost   - When true, ids_ and vals_ reside in host memory (L3 output,
 *            allocated with new[]).  The caller must use delete[] rather than
 *            cudaFree.  When false, both pointers are device pointers.
 */
struct LevelResult {
    SparseGrid* grid = nullptr;
    bool ownsIds = true; //!< False when ids_  is owned by IndexData (L0).
    bool ownsVals = true; //!< False when vals_ is owned by IndexData (L0).
    bool onHost = false; //!< True when the grid lives in host memory (L3).
};

/**
 * @class IQueryStrategy
 * @brief Pure-virtual interface for a per-level memory management strategy.
 *
 * @details A strategy encapsulates the policy for uploading, caching, and freeing
 * index data on the device.  Four concrete implementations exist:
 *
 * L0 - Zero cudaMalloc in the hot path.  All working buffers are pre-allocated
 *      during prepare() and reused across every query and every level.
 *
 * L1 - All data (P vals, radius, maps) permanently resident on the device after
 *      prepare().  Dynamic allocation only for SpTSpM index arrays and output.
 *
 * L2 - Only maps and coarsestMesh are permanently resident.  Radius is uploaded
 *      transiently for pruning and freed immediately after.  P vals are never
 *      uploaded wholesale; only the columns corresponding to pruned centroids
 *      are collected from host memory and uploaded for SpTSpM.
 *
 * L3 - Nothing is permanently resident.  Every kernel invocation is preceded by
 *      a cudaMemGetInfo check; data is uploaded in tiles sized to the available
 *      device memory and freed immediately after each tile.
 *
 * Lifecycle:
 *   1. Constructed with a reference to IndexData.
 *   2. prepare() is called exactly once, before the first query.
 *   3. QueryEngine calls runLevel() once per hierarchy level per query.
 *   4. The destructor releases any strategy-owned device resources.
 */
class IQueryStrategy {
public:
    virtual ~IQueryStrategy() = default;

    /*!
     * @brief Upload and initialise all device-side data required by this strategy.
     *
     * @details Called exactly once before the first query.  Each strategy decides
     * what to upload: L0 pre-allocates all working buffers; L1 uploads all index
     * data permanently; L2/L3 upload only the permanently-resident subset
     * (maps, coarsestMesh) and defer the rest to runLevel().
     *
     * @param[in] reportOs Stream to write the memory policy report to.
     */
    virtual void prepare(std::ostream& reportOs) = 0;

    /*!
     * @brief Execute one hierarchy level: prune -> compact -> SpTSpM -> refactor.
     *
     * @details The caller (QueryEngine) passes the current coarse grid and receives
     * the refined grid for the next level.  Memory management of the returned grid
     * is described by the ownsIds, ownsVals, and onHost fields of LevelResult.
     *
     * @param[in]  l           Level index (0 = coarsest, intervals-1 = finest).
     * @param[in]  currentGrid Input SparseGrid from the previous level.
     * @param[in]  qType       Query type: QueryType::RANGE or QueryType::POINT.
     * @param[in]  lo          Lower bounds of the query box (RANGE queries only).
     * @param[in]  hi          Upper bounds of the query box (RANGE queries only).
     * @param[in]  queryPoint  Query point coordinates (KNN queries only).
     * @param[in]  K           Number of nearest neighbours requested (KNN only).
     * @return LevelResult containing the refined grid and its ownership metadata.
     */
    virtual LevelResult runLevel(size_t l, SparseGrid* currentGrid, QueryType qType, const double* lo, const double* hi,
                                 const double* queryPoint, size_t K) = 0;
};
