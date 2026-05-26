#include "IndexJsonTblaeqQuery.cuh"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "IndexJsonTblaeqJson.cuh"
#include "IndexJsonTblaeqUtils.cuh"
#include "src/core/MemoryPolicy.cuh"
#include "src/core/QueryHandler.cuh"

namespace tblaeq {
namespace file_index {

namespace {
    std::vector<size_t> extract_candidate_indices(const QueryResult& result) {
        if (result.errorCode != 0) throw std::runtime_error(
            "T-BLAEQ query failed with error code " + std::to_string(result.errorCode));
        if (result.fineMesh.empty()) throw std::runtime_error("T-BLAEQ query returned no fine mesh output");
        const SparseGrid* grid = result.fineMesh.front();
        const size_t nnz = grid->numNnz();
        const size_t* ids = grid->ids();
        std::vector<size_t> out;
        out.reserve(nnz);
        for (size_t i = 0; i < nnz; ++i) out.push_back(ids[i]);
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }

    std::string safe_timestamp(double ts) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6) << ts;
        std::string s = oss.str();
        std::replace(s.begin(), s.end(), '.', '_');
        return s;
    }

    std::string extension_for_type(uint8_t dtype) {
        if (dtype == TYPE_IMAGE) return ".png";
        if (dtype == TYPE_LIDAR_360 || dtype == TYPE_LIVOX_AVIA || dtype == TYPE_RADAR_ENCHANCE_PCL) return ".npy";
        if (dtype == TYPE_COORD) return ".coord.bin";
        return ".bin";
    }

    uint64_t checked_add_u64(uint64_t a, uint64_t b, const std::string& ctx) {
        if (a > std::numeric_limits<uint64_t>::max() - b) throw std::runtime_error(ctx);
        return a + b;
    }

    struct PayloadRange {
        fs::path data_path;
        uint64_t abs_start = 0;
        uint64_t size = 0;
    };

    PayloadRange payload_range_checked(const fs::path& index_dir, const IndexRecordFixed& rec, uint8_t slot) {
        if (slot >= rec.slot_count) throw std::runtime_error("slot out of range");
        uint64_t start_rel = rec.data_offsets[slot];
        uint64_t end_rel = rec.data_offsets[slot + 1];
        if (end_rel < start_rel) throw std::runtime_error("invalid data_offsets for slot");
        uint64_t size = end_rel - start_rel;
        uint64_t data_base = rec.index_copy_offset + rec.binary_index_record_size;
        if (data_base < rec.index_copy_offset) throw std::runtime_error("data_base overflow");
        uint64_t abs_start = checked_add_u64(data_base, start_rel, "payload offset overflow");
        fs::path data_path = index_dir / fixed_file_name_to_string(rec.file_name);
        return PayloadRange{std::move(data_path), abs_start, size};
    }

    struct PayloadRequest {
        size_t match_idx = 0;
        uint64_t abs_start = 0;
        uint64_t size = 0;
    };

    struct FileRequests {
        fs::path path;
        uint64_t file_size = 0;
        std::vector<PayloadRequest> requests;
    };

    fs::path build_dump_path(const fs::path& dump_dir, size_t rank, const IndexRecordFixed& rec,
                             const SlotRef& slot_ref) {
        std::ostringstream name;
        name << "match_" << std::setw(4) << std::setfill('0') << rank
            << "_item" << std::setw(8) << std::setfill('0') << rec.item_no
            << "_slot" << std::setw(3) << std::setfill('0') << static_cast<int>(slot_ref.slot)
            << "_seq" << std::setw(4) << std::setfill('0') << static_cast<int>(rec.seq)
            << "_dtype" << static_cast<int>(rec.dtype)
            << "_" << safe_timestamp(slot_ref.timestamp) << extension_for_type(rec.dtype);
        return dump_dir / name.str();
    }

    void dump_payloads(const fs::path& index_dir, const fs::path& dump_dir,
                       const std::vector<IndexRecordFixed>& records,
                       const std::vector<Match>& matches) {
        fs::create_directories(dump_dir);
        if (matches.empty()) return;

        std::vector<fs::path> out_paths(matches.size());
        std::unordered_map<std::string, FileRequests> grouped;
        grouped.reserve(matches.size());

        for (size_t rank = 0; rank < matches.size(); ++rank) {
            const auto& s = matches[rank].slot_ref;
            const auto& rec = records[s.record_idx];
            out_paths[rank] = build_dump_path(dump_dir, rank, rec, s);
            PayloadRange range = payload_range_checked(index_dir, rec, s.slot);
            const std::string key = range.data_path.string();
            auto& group = grouped[key];
            if (group.path.empty()) group.path = range.data_path;
            group.requests.push_back(PayloadRequest{rank, range.abs_start, range.size});
        }

        for (auto& kv : grouped) {
            FileRequests& group = kv.second;
            if (!fs::exists(group.path)) throw std::runtime_error("data file not found: " + group.path.string());
            group.file_size = static_cast<uint64_t>(fs::file_size(group.path));
            for (const auto& req : group.requests) {
                if (req.abs_start > group.file_size || req.size > group.file_size - req.abs_start) {
                    std::ostringstream oss;
                    oss << "payload range exceeds data file length: file=" << group.path
                        << " abs_start=" << req.abs_start << " size=" << req.size
                        << " file_size=" << group.file_size;
                    throw std::runtime_error(oss.str());
                }
            }
        }

        for (auto& kv : grouped) {
            FileRequests& group = kv.second;
            std::ifstream in(group.path, std::ios::binary);
            if (!in) throw std::runtime_error("failed to open data file: " + group.path.string());
            auto& reqs = group.requests;
            std::sort(reqs.begin(), reqs.end(), [](const PayloadRequest& a, const PayloadRequest& b) {
                return a.abs_start < b.abs_start;
            });

            size_t i = 0;
            while (i < reqs.size()) {
                uint64_t span_start = reqs[i].abs_start;
                uint64_t span_end = checked_add_u64(span_start, reqs[i].size, "payload span overflow");
                size_t j = i + 1;
                while (j < reqs.size()) {
                    uint64_t next_start = reqs[j].abs_start;
                    uint64_t next_end = checked_add_u64(next_start, reqs[j].size, "payload span overflow");
                    if (next_start <= span_end) {
                        if (next_end > span_end) span_end = next_end;
                        ++j;
                    }
                    else break;
                }

                uint64_t span_size64 = span_end - span_start;
                if (span_size64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                    throw std::runtime_error("payload span too large to buffer");
                }
                std::vector<uint8_t> buffer(static_cast<size_t>(span_size64));
                if (span_size64 > 0) {
                    in.clear();
                    in.seekg(static_cast<std::streamoff>(span_start), std::ios::beg);
                    in.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
                    if (in.gcount() != static_cast<std::streamsize>(buffer.size())) {
                        throw std::runtime_error("short payload read after length check");
                    }
                }

                for (size_t k = i; k < j; ++k) {
                    const auto& req = reqs[k];
                    fs::path out_path = out_paths[req.match_idx];
                    std::ofstream out(out_path, std::ios::binary);
                    if (!out) throw std::runtime_error("failed to open dump file: " + out_path.string());
                    if (req.size == 0) continue;
                    uint64_t offset64 = req.abs_start - span_start;
                    if (offset64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                        throw std::runtime_error("payload offset too large to buffer");
                    }
                    if (req.size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) - offset64) {
                        throw std::runtime_error("payload size too large to buffer");
                    }
                    size_t offset = static_cast<size_t>(offset64);
                    size_t size = static_cast<size_t>(req.size);
                    if (offset + size > buffer.size()) throw std::runtime_error("payload slice exceeds buffer");
                    out.write(reinterpret_cast<const char*>(buffer.data() + offset),
                              static_cast<std::streamsize>(size));
                }
                i = j;
            }
        }
    }

    void print_matches(const std::vector<IndexRecordFixed>& records, const std::vector<Match>& matches) {
        if (matches.empty()) {
            std::cout << "No matches.\n";
            return;
        }
        for (size_t rank = 0; rank < matches.size(); ++rank) {
            const auto& s = matches[rank].slot_ref;
            const auto& rec = records[s.record_idx];
            uint64_t data_base = rec.index_copy_offset + rec.binary_index_record_size;
            uint64_t abs_offset = data_base + rec.data_offsets[s.slot];
            std::cout << "[" << rank << "] item=" << rec.item_no << " slot=" << static_cast<int>(s.slot) << "\n";
            std::cout << "    seq=" << static_cast<int>(rec.seq)
                << " dtype=" << static_cast<int>(rec.dtype) << "\n";
            std::cout << "    timestamp=" << std::fixed << std::setprecision(9) << s.timestamp
                << " diff_us=" << matches[rank].diff_us
                << " size=" << s.size << "\n";
            std::cout << "    data_file=" << fixed_file_name_to_string(rec.file_name) << "\n";
            std::cout << "    index_copy_offset=" << rec.index_copy_offset << "\n";
            std::cout << "    payload_abs_offset=" << abs_offset << "\n\n";
        }
    }

    std::vector<SlotRef> build_slots_for_records(const std::vector<IndexRecordFixed>& records,
                                                 const std::vector<uint8_t>& seqs,
                                                 bool include_padding) {
        std::vector<SlotRef> slots;
        slots.reserve(records.size() * kGroupSize);
        for (size_t ri = 0; ri < records.size(); ++ri) {
            const auto& rec = records[ri];
            if (!seqs.empty() && !std::binary_search(seqs.begin(), seqs.end(), rec.seq)) continue;
            for (size_t slot = 0; slot < kGroupSize; ++slot) {
                uint64_t size = rec.data_offsets[slot + 1] - rec.data_offsets[slot];
                if (!include_padding && size == 0) continue;
                slots.push_back(SlotRef{
                    ri, static_cast<uint8_t>(slot), rec.timestamp_us[slot], rec.timestamps[slot], size
                });
            }
        }
        std::sort(slots.begin(), slots.end(), [](const SlotRef& a, const SlotRef& b) {
            if (a.timestamp_us != b.timestamp_us) return a.timestamp_us < b.timestamp_us;
            if (a.record_idx != b.record_idx) return a.record_idx < b.record_idx;
            return a.slot < b.slot;
        });
        return slots;
    }

    std::vector<Match> query_interval(const std::vector<SlotRef>& slots, int64_t lo, int64_t hi, size_t top_k) {
        if (slots.empty()) return {};
        auto lower = std::lower_bound(slots.begin(), slots.end(), lo,
                                      [](const SlotRef& s, int64_t v) { return s.timestamp_us < v; });
        auto upper = std::upper_bound(slots.begin(), slots.end(), hi,
                                      [](int64_t v, const SlotRef& s) { return v < s.timestamp_us; });
        std::vector<Match> matches;
        for (auto it = lower; it != upper; ++it) matches.push_back(Match{*it, 0});
        std::sort(matches.begin(), matches.end(), [](const Match& a, const Match& b) {
            if (a.slot_ref.timestamp_us != b.slot_ref.timestamp_us) return a.slot_ref.timestamp_us < b.slot_ref.timestamp_us;
            if (a.slot_ref.record_idx != b.slot_ref.record_idx) return a.slot_ref.record_idx < b.slot_ref.record_idx;
            return a.slot_ref.slot < b.slot_ref.slot;
        });
        if (top_k > 0 && matches.size() > top_k) matches.resize(top_k);
        return matches;
    }

    std::vector<Match> build_matches_knn(const std::vector<IndexRecordFixed>& records,
                                         uint8_t seq,
                                         bool include_padding,
                                         int64_t target,
                                         size_t top_k) {
        std::vector<Match> matches;
        for (size_t ri = 0; ri < records.size(); ++ri) {
            const auto& rec = records[ri];
            if (rec.seq != seq) continue;
            for (size_t slot = 0; slot < kGroupSize; ++slot) {
                uint64_t size = rec.sizes[slot];
                if (!include_padding && size == 0) continue;
                int64_t ts_us = rec.timestamp_us[slot];
                double ts = rec.timestamps[slot];
                int64_t diff = std::llabs(ts_us - target);
                matches.push_back(Match{SlotRef{ri, static_cast<uint8_t>(slot), ts_us, ts, size}, diff});
            }
        }
        std::sort(matches.begin(), matches.end(), [](const Match& a, const Match& b) {
            if (a.diff_us != b.diff_us) return a.diff_us < b.diff_us;
            if (a.slot_ref.timestamp_us != b.slot_ref.timestamp_us) {
                return a.slot_ref.timestamp_us < b.slot_ref.timestamp_us;
            }
            if (a.slot_ref.record_idx != b.slot_ref.record_idx) return a.slot_ref.record_idx < b.slot_ref.record_idx;
            return a.slot_ref.slot < b.slot_ref.slot;
        });
        if (top_k > 0 && matches.size() > top_k) matches.resize(top_k);
        return matches;
    }
} // namespace

int run_query(QueryArgs args) {
    fs::path index_root(args.index_root);
    fs::path index_dir = index_root / args.index_dir;
    fs::path tblaeq_dir = index_root / args.tblaeq_dir;
    if (!fs::exists(index_dir)) throw std::runtime_error("index dir not found: " + index_dir.string());
    if (!fs::exists(tblaeq_dir)) throw std::runtime_error("tblaeq index dir not found: " + tblaeq_dir.string());

    // Query flow: load T-BLAEQ index -> load JSON meta -> coarse query -> refine -> dump payloads.
    QueryHandler handler(tblaeq_dir.string(), /*loadFromIndex=*/true);
    handler.prepareForQuery(LevelPolicy::L3);

    Meta meta = load_meta(index_dir);

    std::vector<uint8_t> seqs = args.seqs;
    if (!seqs.empty()) {
        std::sort(seqs.begin(), seqs.end());
        seqs.erase(std::unique(seqs.begin(), seqs.end()), seqs.end());
    }

    std::vector<size_t> candidates;
    if (args.mode == "knn") {
        if (!args.has_timestamp) throw std::runtime_error("--timestamp is required for knn mode");
        if (!args.has_seq || seqs.size() != 1) {
            throw std::runtime_error("--seq must specify exactly one value for knn mode");
        }
        if (args.knn_k == 0) throw std::runtime_error("--knn-k must be > 0 for knn mode");
        auto v3 = split_timestamp_3d(args.timestamp);
        std::vector<double> query_point = {v3[0], v3[1], v3[2], static_cast<double>(seqs[0])};
        QueryResult result = handler.performSingleKNNQuery(query_point, args.knn_k, /*saveFineMesh=*/true);
        candidates = extract_candidate_indices(result);
    }
    else if (args.mode == "range") {
        if (!args.has_start || !args.has_end) throw std::runtime_error(
            "--start and --end are required for range mode");
        if (!args.has_seq || seqs.empty()) throw std::runtime_error("--seq is required for range mode");
        if (args.end < args.start) throw std::runtime_error("--end must be >= --start");
        auto v3_start = split_timestamp_3d(args.start);
        auto v3_end = split_timestamp_3d(args.end);
        std::vector<double> lower(4);
        std::vector<double> upper(4);
        for (size_t i = 0; i < 3; ++i) {
            lower[i] = std::min(v3_start[i], v3_end[i]);
            upper[i] = std::max(v3_start[i], v3_end[i]);
        }
        lower[3] = static_cast<double>(seqs.front());
        upper[3] = static_cast<double>(seqs.back());
        QueryResult result = handler.performSingleRangeQuery(upper, lower, /*saveFineMesh=*/true);
        candidates = extract_candidate_indices(result);
    }
    else {
        throw std::runtime_error("unsupported mode: " + args.mode);
    }

    std::cout << "Coarse candidates: " << candidates.size() << "\n";
    if (candidates.empty()) {
        std::cout << "Loaded candidate records: 0\n";
        std::cout << "Matches: 0\n";
        return 0;
    }

    std::vector<IndexRecordFixed> records = load_candidate_records(index_dir, meta, candidates);
    std::vector<Match> matches;
    if (args.mode == "knn") {
        int64_t target = timestamp_to_us(args.timestamp);
        matches = build_matches_knn(records, seqs[0], args.include_padding, target, args.knn_k);
    }
    else {
        int64_t lo = timestamp_to_us(args.start);
        int64_t hi = timestamp_to_us(args.end);
        std::vector<SlotRef> slots = build_slots_for_records(records, seqs, args.include_padding);
        matches = query_interval(slots, lo, hi, args.range_limit);
        std::cout << "Fine slots: " << slots.size() << "\n";
    }

    std::cout << "Loaded candidate records: " << records.size() << "\n";
    std::cout << "Matches: " << matches.size() << "\n";
    print_matches(records, matches);

    fs::path dump_dir = args.has_dump_dir ? fs::path(args.dump_dir) : (index_root / "payloads");
    dump_payloads(index_dir, dump_dir, records, matches);
    std::cout << "Dumped " << matches.size() << " payload(s) to: " << dump_dir << "\n";
    return 0;
}

} // namespace file_index
} // namespace tblaeq
