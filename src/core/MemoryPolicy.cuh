/**
 * @file MemoryPolicy.cuh
 * @brief Memory policy types and the PolicyScheduler that assigns them.
 *
 * @details This file defines the four memory policies (L0-L3), the per-level
 * policy container IndexPolicy, and the PolicyScheduler that selects the most
 * aggressive policy each level can afford given the available device memory.
 *
 * The scheduler uses a simple greedy algorithm: it iterates levels from
 * coarsest to finest and assigns the best policy that fits within the
 * remaining budget.  L0 and L1 consume budget permanently; L2 is transient
 * and does not reduce the long-term budget; L3 requires no device memory at
 * schedule time.
 */

#pragma once

#include <cstddef>
#include <iosfwd>
#include <memory>
#include <vector>

struct IndexData;
class IQueryStrategy;

/**
 * @enum LevelPolicy
 * @brief Memory management strategy assigned to a single hierarchy level.
 *
 * @details The four policies form a spectrum from maximum performance (L0)
 * to minimum device memory consumption (L3):
 *
 * L0 - Zero cudaMalloc in the query hot path. All working buffers (mask,
 *      clusters, compact output, SpTSpM index arrays, yValue) are
 *      pre-allocated during prepare() and reused across every query.
 *      P vals and radius are permanently cached on the device.
 *
 * L1 - P vals and radius permanently cached. SpTSpM index arrays and output
 *      are allocated dynamically per query, but no pre-allocation overhead.
 *
 * L2 - Only maps and coarsestMesh are permanently resident. Radius is
 *      uploaded transiently before pruning and freed immediately after.
 *      SpTSpM uploads only the P-tensor columns corresponding to the pruned
 *      centroids, avoiding the cost of transferring the full P vals array.
 *
 * L3 - Unconditional tiling fallback. Nothing is permanently resident.
 *      Every operation queries cudaMemGetInfo and processes data in tiles
 *      sized to the available device memory.  The output grid lives in host
 *      memory (LevelResult::onHost == true).
 */
enum class LevelPolicy { L0 = 0, L1 = 1, L2 = 2, L3 = 3 };

/*!
 * @brief Return the human-readable name of a LevelPolicy value.
 *
 * @param[in] p The policy to name.
 * @return A null-terminated string such as "L0", "L1", "L2", or "L3".
 */
const char* levelPolicyName(LevelPolicy p);

/**
 * @struct IndexPolicy
 * @brief Complete per-level policy assignment for one IndexData instance.
 *
 * @details Stores one LevelPolicy per hierarchy level and provides helpers
 * used by strategies and the scheduler.
 */
struct IndexPolicy {
    std::vector<LevelPolicy> levels; //!< levels[l] = policy assigned to level l.

    LevelPolicy operator[](size_t l) const { return levels[l]; }

    /*!
     * @brief Return true when at least one level uses L0.
     *
     * @details Used by strategies to decide whether to call allocSpTSpMBuffers().
     * @return True if any level is LevelPolicy::L0.
     */
    bool anyL0() const;

    /*!
     * @brief Return true when at least one level uses L2 or L3.
     *
     * @return True if any level is LevelPolicy::L2 or LevelPolicy::L3.
     */
    bool anyStreaming() const;

    /*!
     * @brief Compute the additional device bytes consumed by L0 pre-allocation.
     *
     * @details Sums the SpTSpM buffer sizes for all L0 levels.
     * @param[in] idx The index whose P-tensor sizes are used for the calculation.
     * @return Total additional device bytes for L0 SpTSpM buffers.
     */
    size_t l0ExtraBytes(const IndexData& idx) const;

    /*!
     * @brief Construct a uniform policy where every level uses the same strategy.
     *
     * @details Useful for benchmarking and forced policy testing via --force-policy.
     * @param[in] n Number of levels.
     * @param[in] p The policy to assign to all levels.
     * @return An IndexPolicy with n levels all set to p.
     */
    static IndexPolicy uniform(size_t n, LevelPolicy p) {
        IndexPolicy pol;
        pol.levels.assign(n, p);
        return pol;
    }

    /*!
     * @brief Parse a policy name string into a LevelPolicy value.
     *
     * @param[in] s One of "L0", "L1", "L2", "L3" (case-insensitive).
     * @return The corresponding LevelPolicy enum value.
     * @throws std::invalid_argument When s does not match any known policy name.
     */
    static LevelPolicy parseLevel(const std::string& s);
};

/**
 * @struct DeviceMemoryInfo
 * @brief Snapshot of current device memory availability.
 */
struct DeviceMemoryInfo {
    size_t freeMem = 0; //!< Currently free device memory in bytes.
    size_t totalMem = 0; //!< Total device memory in bytes.

    /*!
     * @brief Query the current device memory state via cudaMemGetInfo.
     * @return A DeviceMemoryInfo populated with the current free and total values.
     */
    static DeviceMemoryInfo query();
};

/**
 * @class PolicyScheduler
 * @brief Assigns the most aggressive feasible memory policy to each level.
 *
 * @details The scheduler uses a greedy algorithm over levels ordered from
 * coarsest (l=0) to finest (l=intervals-1):
 *
 *   budget = free_device_mem * kSafetyFactor - permanent_overhead
 *
 *   For each level l:
 *     if   budget >= cost_L0  ->  assign L0, subtract cost_L0 from budget
 *     elif budget >= cost_L1  ->  assign L1, subtract cost_L1 from budget
 *     elif budget >= cost_L2  ->  assign L2, budget unchanged (transient)
 *     else                    ->  assign L3, budget unchanged
 *
 * L0 and L1 reduce the remaining budget because their data stays on the
 * device for the lifetime of the index.  L2 peak memory is transient
 * (radius freed after pruning; only selected P columns uploaded for SpTSpM),
 * so it does not consume long-term budget.  L3 manages everything in tiles
 * and requires no reserved device memory at schedule time.
 */
class PolicyScheduler {
public:
    static constexpr double kSafetyFactor = 0.85; //!< Fraction of free memory used as budget.

    /*!
     * @brief Compute and return the per-level IndexPolicy for the given index.
     *
     * @details Calls cudaMemGetInfo once to determine the available budget, then
     * applies the greedy algorithm described in the class documentation.
     * IndexData::computeStats() must have been called before this function.
     *
     * @param[in] idx The index whose memory footprint drives the decision.
     * @return The recommended IndexPolicy with one LevelPolicy per level.
     */
    static IndexPolicy recommend(const IndexData& idx);

    /*!
     * @brief Print a human-readable per-level memory policy report to a stream.
     *
     * @param[in] idx    The index whose stats are reported.
     * @param[in] policy The policy to describe.
     * @param[in] os     Output stream (e.g. std::cout).
     */
    static void printReport(const IndexData& idx, const IndexPolicy& policy, std::ostream& os);

    /*!
     * @brief Create the prepared strategy object for a given IndexPolicy.
     *
     * @details When all levels share the same LevelPolicy a single strategy
     * instance handles all levels. The returned object has NOT been prepared;
     * the caller must invoke strategy->prepare() before running queries.
     *
     * @param[in,out] idx    The index the strategy will operate on.
     * @param[in]     policy The per-level policy assignment.
     * @return Owning pointer to the newly created (unprepared) strategy.
     */
    static std::unique_ptr<IQueryStrategy> make(IndexData& idx, const IndexPolicy& policy);

    /*!
     * @brief Create a single-level strategy for level l.
     *
     * @details Called by QueryEngine::runSingleQuery() to dispatch per-level
     * execution when the active policy assigns different strategies to
     * different levels.  The returned strategy is not prepared; it is used
     * only for its runLevel() method.
     *
     * @param[in,out] idx The index the strategy will operate on.
     * @param[in]     l   Level index (0 = coarsest).
     * @return Owning pointer to the strategy for level l.
     */
    static std::unique_ptr<IQueryStrategy> makeForLevel(IndexData& idx, size_t l);
};
