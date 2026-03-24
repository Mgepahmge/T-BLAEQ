/**
 * @file QueryEngine.cuh
 * @brief Outer query loop, result collection, and per-level strategy dispatch.
 *
 * @details QueryEngine owns the structure of a query batch: it iterates over
 * individual queries, constructs the coarsestView grid, manages the lifetime
 * of inter-level SparseGrid objects, collects timing, and delegates all
 * per-level computation to the active IQueryStrategy.
 *
 * The separation of concerns is strict: QueryEngine knows nothing about how
 * data is uploaded or freed; that is entirely the responsibility of the
 * strategy.  QueryEngine only needs to free the grid returned by each
 * runLevel() call, using the ownsIds, ownsVals, and onHost flags in
 * LevelResult to determine the correct deallocation path.
 */

#pragma once

#include <limits>
#include <memory>
#include <string>
#include <vector>
#include "IndexData.cuh"
#include "Memory.cuh"
#include "MemoryPolicy.cuh"
#include "src/Data_Structures/File.cuh"
#include "strategies/IQueryStrategy.cuh"
#include "strategies/StrategyCommon.cuh"

/**
 * @struct QueryResult
 * @brief Aggregated output of a query batch execution.
 *
 * @details Owns the fine-mesh SparseGrid pointers when saveFineMesh is true.
 * The destructor releases them correctly according to the ownsIds, ownsVals,
 * and onHost flags stored in parallel vectors.
 */
struct QueryResult {
    ~QueryResult();

    QueryType type = QueryType::RANGE; //!< Range or KNN query.
    int errorCode = 0; //!< Non-zero on execution error.
    long totalTimeUs = 0; //!< Total execution time in microseconds.
    size_t queryCount = 0; //!< Number of queries executed.

    std::string datasetName; //!< Name of the dataset used.
    std::string queryParam; //!< Human-readable query parameter string.
    size_t datasetSize = 0; //!< Number of points in the dataset.
    size_t datasetDim = 0; //!< Dimensionality of the dataset.

    std::vector<double> queryRangeVolume; //!< Per-query range volume (RANGE queries).
    std::vector<size_t> fineMeshSize; //!< Per-query fine-mesh point count.
    std::vector<SparseGrid*> fineMesh; //!< Per-query fine-mesh grids (when saveFineMesh).
    std::vector<bool> fineMeshOwnsIds; //!< Ownership flag for ids_ of each fine mesh.
    std::vector<bool> fineMeshOnHost; //!< True when the fine mesh is host-resident (L3).
};

/**
 * @class QueryEngine
 * @brief Executes query batches by iterating queries and dispatching per-level work.
 *
 * @details The engine constructs a coarsestView SparseGrid backed by the permanent
 * device buffers dCoarsestMeshIds/Vals, then calls the strategy's runLevel() for
 * each hierarchy level.  Between levels it frees the previous grid using the
 * ownership metadata in LevelResult, preventing both leaks and double-frees.
 */
class QueryEngine {
public:
    /**
     * @struct RunConfig
     * @brief Execution parameters for a query batch.
     */
    struct RunConfig {
        bool saveFineMesh = false; //!< When true, fine-mesh grids are retained in QueryResult.
        int maxQueryCount = std::numeric_limits<int>::max(); //!< Maximum number of queries to execute.
        size_t K = 0; //!< Number of nearest neighbours (KNN queries).
    };

    /*!
     * @brief Execute a query batch using the already-prepared strategy.
     *
     * @details The strategy must have been fully prepared (strategy.prepare()
     * called) before this function is invoked.  QueryHandler is responsible
     * for creating and preparing the strategy; QueryEngine only executes.
     *
     * For each query the engine calls runSingleQuery(), which in turn calls
     * strategy.runLevel() for each hierarchy level via PolicyScheduler::makeForLevel().
     *
     * @param[in,out] idx       Fully built and prepared IndexData.
     * @param[in,out] strategy  Prepared strategy object.
     * @param[in]     queryData Pre-loaded query batch.
     * @param[in]     cfg       Execution parameters.
     * @return Aggregated results for all executed queries.
     */
    static QueryResult run(IndexData& idx, IQueryStrategy& strategy, const Query& queryData, const RunConfig& cfg);

private:
    /*!
     * @brief Execute the full hierarchy pipeline for a single query.
     *
     * @details Constructs the initial coarsestView grid, then iterates levels
     * calling PolicyScheduler::makeForLevel() to obtain the per-level strategy
     * and invoking its runLevel().  Between levels the previous grid is freed
     * using the LevelResult ownership flags to select the correct path
     * (cudaFree vs delete[] vs no-op).
     *
     * @param[in,out] idx        IndexData containing index and device buffers.
     * @param[in]     qType      Query type (RANGE or KNN).
     * @param[in]     lo         Lower bounds of the query box (RANGE only).
     * @param[in]     hi         Upper bounds of the query box (RANGE only).
     * @param[in]     queryPoint Query point coordinates (KNN only).
     * @param[in]     K          Number of nearest neighbours (KNN only).
     * @return LevelResult from the finest level, representing the query output.
     */
    static LevelResult runSingleQuery(IndexData& idx, QueryType qType, const double* lo, const double* hi,
                                      const double* queryPoint, size_t K);
};

/*!
 * @brief Append query statistics to a CSV file.
 *
 * @param[in] result     The query result to serialise.
 * @param[in] outputFile Path to the output CSV file (appended if it exists).
 */
void saveQueryResult(const QueryResult& result, const std::string& outputFile);
