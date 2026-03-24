#include <cuda_runtime.h>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "IndexData.cuh"
#include "MemoryPolicy.cuh"
#include "src/Query/check.cuh"

#include "strategies/L0Strategy.cuh"
#include "strategies/L1Strategy.cuh"
#include "strategies/L2Strategy.cuh"
#include "strategies/L3Strategy.cuh"

static std::string fmtBytes(size_t b) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(2);
    if (b >= (1ULL << 30)) {
        o << b / double(1ULL << 30) << " GiB";
    }
    else if (b >= (1ULL << 20)) {
        o << b / double(1ULL << 20) << " MiB";
    }
    else if (b >= (1ULL << 10)) {
        o << b / double(1ULL << 10) << " KiB";
    }
    else {
        o << b << " B";
    }
    return o.str();
}

const char* levelPolicyName(LevelPolicy p) {
    switch (p) {
    case LevelPolicy::L0:
        return "L0";
    case LevelPolicy::L1:
        return "L1";
    case LevelPolicy::L2:
        return "L2";
    case LevelPolicy::L3:
        return "L3";
    default:
        return "?";
    }
}

LevelPolicy IndexPolicy::parseLevel(const std::string& s) {
    if (s == "L0" || s == "l0") {
        return LevelPolicy::L0;
    }
    if (s == "L1" || s == "l1") {
        return LevelPolicy::L1;
    }
    if (s == "L2" || s == "l2") {
        return LevelPolicy::L2;
    }
    if (s == "L3" || s == "l3") {
        return LevelPolicy::L3;
    }
    throw std::invalid_argument("Unknown policy level '" + s + "'. Valid values: L0, L1, L2, L3.");
}

bool IndexPolicy::anyL0() const {
    for (auto p : levels) {
        if (p == LevelPolicy::L0) {
            return true;
        }
    }
    return false;
}

bool IndexPolicy::anyStreaming() const {
    for (auto p : levels) {
        if (p == LevelPolicy::L2 || p == LevelPolicy::L3) {
            return true;
        }
    }
    return false;
}

size_t IndexPolicy::l0ExtraBytes(const IndexData& idx) const {
    size_t total = 0;
    for (size_t l = 0; l < levels.size(); ++l) {
        if (levels[l] == LevelPolicy::L0) {
            total += 3 * idx.pTensors[l]->get_row_nums() * sizeof(size_t);
        }
    }
    return total;
}

DeviceMemoryInfo DeviceMemoryInfo::query() {
    DeviceMemoryInfo info;
    CUDA_CHECK(cudaMemGetInfo(&info.freeMem, &info.totalMem));
    return info;
}

IndexPolicy PolicyScheduler::recommend(const IndexData& idx) {
    const auto mem = DeviceMemoryInfo::query();

    // Start with the safe-factored free memory.
    size_t budget = static_cast<size_t>(mem.freeMem * kSafetyFactor);

    // Permanent overhead that must always be on device regardless of policy:
    //   - coarsestMesh ids + vals
    //   - all dMaps  (one per level, size = P[l].row_nums)
    //   - pTensors[l]->get_nnz_per_col() stays on host (KNN STEP only)
    const size_t permanentCost = idx.stats.deviceCoarsestMesh + idx.stats.deviceMaps;

    if (budget <= permanentCost) {
        // Cannot even fit the permanent overhead -- assign L3 to everything.
        IndexPolicy policy;
        policy.levels.assign(idx.intervals, LevelPolicy::L3);
        return policy;
    }
    budget -= permanentCost;

    // Assign policies level by level, coarsest (l=0) first.
    IndexPolicy policy;
    policy.levels.resize(idx.intervals);

    for (size_t l = 0; l < idx.intervals; ++l) {
        const size_t nnz = idx.pTensors[l]->get_nnz_nums();
        const size_t col = idx.pTensors[l]->get_col_nums();
        const size_t D = idx.D;

        const size_t pValsBytes = nnz * D * sizeof(double);
        const size_t radiusBytes = col * sizeof(double);
        const size_t spBufBytes = 3 * nnz * sizeof(size_t);

        // Peak memory during SpTSpM execution (conservative upper bounds):
        //   yValue output       = numProcessedNonZero * D * 8B <= pValsBytes
        //   SpTSpM tmp buffers  = spBufBytes  (3 index arrays, freed after kernel)
        // numProcessedNonZero in practice << nnz (pruning removes most centroids),
        // but we use nnz as a safe upper bound since we don't know selectCount at scheduling time.
        const size_t yValuePeak = pValsBytes; // conservative: full P.nnz * D
        const size_t tmpPeak = spBufBytes; // 3 index arrays during SpTSpM

        // L0: permanent resident = pVals + radius + spBuf (pre-alloc)
        //     transient peak     = yValue (still one cudaMalloc for output)
        const size_t costL0 = pValsBytes + radiusBytes + spBufBytes + yValuePeak;

        // L1: permanent resident = pVals + radius
        //     transient peak     = SpTSpM tmp + yValue (both freed after level)
        const size_t costL1 = pValsBytes + radiusBytes + tmpPeak + yValuePeak;

        // L2: "unblocked L3" -- lazy load everything except permanent data.
        //     Radius is uploaded for pruning then freed immediately.
        //     Peak occurs during SpTSpM:
        //       P[l].vals (uploaded whole) + yValue + spBuf + pruned.vals
        //     pruned.vals <= col*D*8 is dominated by pValsBytes, so:
        //     peak ~ 2*pVals + spBuf  (radius already freed by SpTSpM phase)
        //     Does NOT reduce long-term budget (all freed before next level).
        const size_t costL2 = 2 * pValsBytes + spBufBytes;

        if (budget >= costL0) {
            policy.levels[l] = LevelPolicy::L0;
            budget -= costL0;
        }
        else if (budget >= costL1) {
            policy.levels[l] = LevelPolicy::L1;
            budget -= costL1;
        }
        else if (budget >= costL2) {
            // L2: transient -- does not reduce long-term budget.
            policy.levels[l] = LevelPolicy::L2;
            // budget unchanged
        }
        else {
            policy.levels[l] = LevelPolicy::L3;
        }
    }

    return policy;
}

void PolicyScheduler::printReport(const IndexData& idx, const IndexPolicy& policy, std::ostream& os) {
    const auto mem = DeviceMemoryInfo::query();
    const size_t available = static_cast<size_t>(mem.freeMem * kSafetyFactor);

    os << "\n=== Per-Level Memory Policy Report ===\n";
    os << "  Device total   : " << fmtBytes(mem.totalMem) << "\n";
    os << "  Device free    : " << fmtBytes(mem.freeMem) << "\n";
    os << "  Budget (" << static_cast<int>(kSafetyFactor * 100) << "%): " << fmtBytes(available) << "\n\n";

    os << "  Permanent overhead:\n";
    os << "    coarsestMesh  : " << fmtBytes(idx.stats.deviceCoarsestMesh) << "\n";
    os << "    maps (all)    : " << fmtBytes(idx.stats.deviceMaps) << "\n";


    struct ReportRow {
        std::string level;
        std::string pVals;
        std::string radius;
        std::string peakCost;
        std::string policyName;
    };

    const std::string hLevel = "Level";
    const std::string hPVals = "P vals";
    const std::string hRadius = "radius";
    const std::string hPeak = "peak cost";
    const std::string hPolicy = "Policy";

    std::vector<ReportRow> rows;
    rows.reserve(idx.intervals);

    size_t wLevel = hLevel.size();
    size_t wPVals = hPVals.size();
    size_t wRadius = hRadius.size();
    size_t wPeak = hPeak.size();
    size_t wPolicy = hPolicy.size();

    for (size_t l = 0; l < idx.intervals; ++l) {
        const size_t nnz = idx.pTensors[l]->get_nnz_nums();
        const size_t col = idx.pTensors[l]->get_col_nums();
        const size_t D = idx.D;
        const size_t pv = nnz * D * sizeof(double);
        const size_t rv = col * sizeof(double);
        const size_t sp = 3 * nnz * sizeof(size_t);
        const LevelPolicy lp = policy.levels[l];
        // peak cost = what must fit in device memory simultaneously at SpTSpM time
        const size_t peak = (lp == LevelPolicy::L0) ? pv + rv + sp + pv : // pVals(resident)+radius+spBuf+yValue
            (lp == LevelPolicy::L1) ? pv + rv + sp + pv
                                    : // same: pVals resident, tmp+yValue transient
            pv + rv + sp + pv; // L2/L3: all transient, same peak

        ReportRow row{std::to_string(l), fmtBytes(pv), fmtBytes(rv), fmtBytes(peak), levelPolicyName(lp)};
        wLevel = std::max(wLevel, row.level.size());
        wPVals = std::max(wPVals, row.pVals.size());
        wRadius = std::max(wRadius, row.radius.size());
        wPeak = std::max(wPeak, row.peakCost.size());
        wPolicy = std::max(wPolicy, row.policyName.size());
        rows.push_back(std::move(row));
    }

    constexpr size_t gap = 2;
    const std::string sep(gap, ' ');
    const size_t tableWidth = wLevel + wPVals + wRadius + wPeak + wPolicy + (gap * 4);
    const int iwLevel = static_cast<int>(wLevel);
    const int iwPVals = static_cast<int>(wPVals);
    const int iwRadius = static_cast<int>(wRadius);
    const int iwPeak = static_cast<int>(wPeak);
    const int iwPolicy = static_cast<int>(wPolicy);

    os << "  " << std::left << std::setw(iwLevel) << hLevel << sep << std::setw(iwPVals) << hPVals << sep
       << std::setw(iwRadius) << hRadius << sep << std::setw(iwPeak) << hPeak << sep << std::setw(iwPolicy)
       << hPolicy << "\n";
    os << "  " << std::string(tableWidth, '-') << "\n";

    for (const auto& row : rows) {
        os << "  " << std::right << std::setw(iwLevel) << row.level << sep << std::setw(iwPVals) << row.pVals
           << sep << std::setw(iwRadius) << row.radius << sep << std::setw(iwPeak) << row.peakCost << sep
           << std::left << std::setw(iwPolicy) << row.policyName << "\n";
    }

    if (policy.anyL0()) {
        os << "\n  L0 extra total : " << fmtBytes(policy.l0ExtraBytes(idx)) << "\n";
    }
    if (policy.anyStreaming()) {
        os << "  [L2/L3 levels will upload/free P vals per query]\n";
    }
    os << "======================================\n\n";
}

std::unique_ptr<IQueryStrategy> PolicyScheduler::make(IndexData& idx, const IndexPolicy& policy) {
    idx.activePolicy = policy;
    // When all levels share the same policy, return that single strategy type.
    // Mixed-policy IndexData is fully supported: QueryEngine calls makeForLevel()
    // per level, but the "make" entry point is used by QueryHandler which creates
    // one representative strategy for the prepare() call.
    // For a mixed policy, use the finest (last) level's policy as representative,
    // since that level dominates memory usage.
    const LevelPolicy rep = policy.levels.empty() ? LevelPolicy::L1 : policy.levels.back();
    switch (rep) {
    case LevelPolicy::L0:
        return std::make_unique<L0Strategy>(idx);
    case LevelPolicy::L1:
        return std::make_unique<L1Strategy>(idx);
    case LevelPolicy::L2:
        return std::make_unique<L2Strategy>(idx);
    case LevelPolicy::L3:
        return std::make_unique<L3Strategy>(idx);
    default:
        return std::make_unique<L1Strategy>(idx);
    }
}

std::unique_ptr<IQueryStrategy> PolicyScheduler::makeForLevel(IndexData& idx, size_t l) {
    switch (idx.activePolicy.levels[l]) {
    case LevelPolicy::L0:
        return std::make_unique<L0Strategy>(idx);
    case LevelPolicy::L1:
        return std::make_unique<L1Strategy>(idx);
    case LevelPolicy::L2:
        return std::make_unique<L2Strategy>(idx);
    case LevelPolicy::L3:
        return std::make_unique<L3Strategy>(idx);
    default:
        return std::make_unique<L1Strategy>(idx);
    }
}
