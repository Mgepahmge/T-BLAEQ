/**
 * @file QueryHandler.cuh
 * @brief Public facade for building, loading, and querying the T-BLAEQ index.
 *
 * @details QueryHandler is the top-level entry point for all user-facing
 * operations.  It owns the IndexData and the active IQueryStrategy, and
 * orchestrates the prepare-then-query lifecycle.
 *
 * Typical usage:
 *   QueryHandler h("indexes/sift/", true);  // load from disk
 *   h.prepareForQuery();                     // auto-select policy
 *   auto r = h.performQuery("queries.txt", QueryType::RANGE);
 *   saveQueryResult(r, "out.csv");
 */

#pragma once

#include <iosfwd>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include "IndexBuilder.cuh"
#include "IndexData.cuh"
#include "IndexSerializer.cuh"
#include "MemoryPolicy.cuh"
#include "QueryEngine.cuh"
#include "src/Data_Structures/File.cuh"
#include "strategies/IQueryStrategy.cuh"

/**
 * @class QueryHandler
 * @brief Public facade for the T-BLAEQ index: build/load, prepare, and query.
 *
 * @details QueryHandler encapsulates the full lifecycle of the index:
 *
 * Construction - either builds a new index from a raw dataset path, or loads
 * a previously serialised index from disk.
 *
 * Preparation - prepareForQuery() selects a per-level memory policy via
 * PolicyScheduler::recommend() (or accepts a manual override), creates the
 * matching IQueryStrategy, calls strategy_->prepare() to upload data to the
 * device, and sets prepared_ = true.  Subsequent calls to prepareForQuery()
 * are no-ops.
 *
 * Query execution - performQuery() loads queries from a file and delegates to
 * QueryEngine::run().  It calls ensurePrepared() first, so explicit preparation
 * is optional.
 */
class QueryHandler {
public:
    /*!
     * @brief Build a new index from a raw dataset on disk.
     *
     * @param[in] datasetPath Path to the dataset file used to build the index.
     * @param[in] forceUseCPU
     * @param[in] height  Total number of mesh levels.
     * @param[in] ratios  Coarsening ratio per level; length must equal height - 1.
     */
    explicit QueryHandler(bool forceUseCPU, const std::string& datasetPath,
                          size_t height = IndexBuilder::kDefaultHeight,
                          const std::vector<size_t>& ratios = IndexBuilder::kDefaultRatios);

    /*!
     * @brief Build an index from randomly generated synthetic data.
     *
     * @details Uses IndexBuilder::buildRandom() to generate a synthetic
     * multi-level dataset instantly without loading or clustering real data.
     * Intended for rapid testing of the query pipeline.
     *
     * @param[in] N       Number of points at the finest mesh level.
     * @param[in] D       Dimensionality of each point.
     * @param[in] valMin  Lower bound of the generated value range (must be > 0).
     * @param[in] valMax  Upper bound of the generated value range.
     * @param[in] isInt   When true, all generated coordinates are integers.
     * @param[in] height  Total number of mesh levels.
     * @param[in] ratios  Coarsening ratio per level; length must equal height - 1.
     * @param[in] seed    Random seed used by RandomKmeans.
     * @param[in] sigmaDivisor
     *                    Controls random spread per level: sigma = spacing / sigmaDivisor.
     * @param[in] name    Human-readable label stored in IndexData for logging.
     */
    QueryHandler(size_t N, size_t D, double valMin, double valMax, bool isInt,
                 size_t height = IndexBuilder::kDefaultHeight,
                 const std::vector<size_t>& ratios = IndexBuilder::kDefaultRatios,
                 uint64_t seed = 12345, double sigmaDivisor = 3.0,
                 const std::string& name = "random");

    /*!
     * @brief Load a previously serialised index from disk.
     *
     * @param[in] indexPath      Directory path containing the serialised index files.
     * @param[in] loadFromIndex  Must be true; distinguishes this overload from the
     *                           build constructor.
     */
    QueryHandler(const std::string& indexPath, bool loadFromIndex);
    ~QueryHandler() = default;

    QueryHandler(const QueryHandler&) = delete;
    QueryHandler& operator=(const QueryHandler&) = delete;

    /*!
     * @brief Serialise the current index to disk.
     *
     * @param[in] dirPath Directory path to write the index files into.
     */
    void saveIndex(const std::string& dirPath) const;

    /*!
     * @brief Auto-select the per-level policy and prepare the strategy.
     *
     * @details Calls PolicyScheduler::recommend() to choose the most aggressive
     * policy that fits in device memory, then calls doPrepare(). Idempotent:
     * subsequent calls are no-ops once prepared_ is true.
     */
    void prepareForQuery();

    /*!
     * @brief Prepare the strategy using a manually specified per-level policy.
     *
     * @details Bypasses the automatic scheduler and uses the provided policy
     * directly. Idempotent once prepared_ is true.
     *
     * @param[in] policy The per-level policy to apply.
     */
    void prepareForQuery(IndexPolicy policy);

    /*!
     * @brief Force all levels to use the same policy and prepare.
     *
     * @details Constructs a uniform IndexPolicy with the given level policy
     * applied to every level, then calls doPrepare(). Useful for benchmarking
     * and forced policy testing. Idempotent once prepared_ is true.
     *
     * @param[in] level The LevelPolicy to assign to all levels.
     */
    void prepareForQuery(LevelPolicy level);

    /*!
     * @brief Print a human-readable memory statistics table.
     *
     * @note Valid only after prepareForQuery() has been called.
     * @param[in] os Output stream to write to.
     */
    void printMemoryStats(std::ostream& os) const;

    /*!
     * @brief Return the memory consumption snapshot computed during prepare.
     *
     * @return Const reference to the MemoryStats stored in IndexData.
     */
    const MemoryStats& getMemoryStats() const { return idx_->stats; }

    /*!
     * @brief Return the active per-level policy assignment.
     *
     * @return Const reference to the IndexPolicy stored in IndexData.
     */
    const IndexPolicy& getActivePolicy() const { return idx_->activePolicy; }

    /*!
     * @brief Load queries from file and execute the hierarchical query pipeline.
     *
     * @details Calls ensurePrepared() first, so explicit preparation is optional.
     * Delegates execution to QueryEngine::run() after loading the query file.
     *
     * @param[in] queryPath     Path to the query file.
     * @param[in] qType         QueryType::RANGE or QueryType::POINT.
     * @param[in] saveFineMesh  When true, fine-mesh SparseGrids are retained in
     *                          QueryResult::fineMesh for inspection.
     * @param[in] maxQueryCount Maximum number of queries to execute.
     * @param[in] K             Number of nearest neighbours (KNN queries only).
     * @return Aggregated results for all executed queries.
     */
    QueryResult performQuery(const std::string& queryPath, QueryType qType, bool saveFineMesh = false,
                             int maxQueryCount = std::numeric_limits<int>::max(), size_t K = 0);

    /*!
     * @brief Return the number of data points in the index.
     * @return Total point count N.
     */
    [[nodiscard]] size_t getSize() const { return idx_->N; }

    /*!
     * @brief Return the dimensionality of the indexed data.
     * @return Dimensionality D.
     */
    [[nodiscard]] size_t getDim() const { return idx_->D; }

private:
    std::unique_ptr<IndexData> idx_; //!< Owned index data container.
    std::unique_ptr<IQueryStrategy> strategy_; //!< Owned strategy, valid after prepare.
    bool prepared_ = false; //!< True after the first successful prepare.

    /*!
     * @brief Create the strategy for the given policy, call prepare(), set prepared_.
     *
     * @param[in] policy The per-level policy to use.
     * @param[in] label  Human-readable label printed to std::cout before prepare.
     */
    void doPrepare(const IndexPolicy& policy, const std::string& label);

    /*!
     * @brief Call prepareForQuery() if not already prepared.
     */
    void ensurePrepared();
};
