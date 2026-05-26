#include "IndexJsonTblaeqJson.cuh"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "IndexJsonTblaeqPicojson.cuh"
#include "IndexJsonTblaeqUtils.cuh"

namespace tblaeq {
namespace file_index {

namespace {
    const picojson::object& require_object(const picojson::value& v, const std::string& ctx) {
        if (!v.is<picojson::object>()) throw std::runtime_error("JSON root must be object: " + ctx);
        return v.get<picojson::object>();
    }

    const picojson::value& require_field(const picojson::object& obj, const std::string& key) {
        auto it = obj.find(key);
        if (it == obj.end()) throw std::runtime_error("missing JSON field: " + key);
        return it->second;
    }

    std::string json_string(const picojson::object& obj, const std::string& key) {
        const picojson::value& v = require_field(obj, key);
        if (!v.is<std::string>()) throw std::runtime_error("JSON field must be string: " + key);
        return v.get<std::string>();
    }

    double json_double_value(const picojson::value& v, const std::string& key) {
        if (!v.is<double>()) throw std::runtime_error("JSON field must be number: " + key);
        return v.get<double>();
    }

    double json_double(const picojson::object& obj, const std::string& key) {
        return json_double_value(require_field(obj, key), key);
    }

    int64_t json_i64_value(const picojson::value& v, const std::string& key) {
#ifdef PICOJSON_USE_INT64
        if (v.is<int64_t>()) return v.get<int64_t>();
#endif
        if (!v.is<double>()) throw std::runtime_error("JSON field must be integer number: " + key);
        double d = v.get<double>();
        if (!std::isfinite(d) || d < static_cast<double>(std::numeric_limits<int64_t>::min()) ||
            d > static_cast<double>(std::numeric_limits<int64_t>::max())) {
            throw std::runtime_error("JSON integer out of int64 range: " + key);
        }
        double rounded = std::round(d);
        if (std::abs(d - rounded) > 0.0) throw std::runtime_error("JSON field is not integral: " + key);
        return static_cast<int64_t>(rounded);
    }

    int64_t json_i64(const picojson::object& obj, const std::string& key) {
        return json_i64_value(require_field(obj, key), key);
    }

    uint64_t json_u64(const picojson::object& obj, const std::string& key) {
        int64_t v = json_i64(obj, key);
        if (v < 0) throw std::runtime_error("JSON field must be non-negative: " + key);
        return static_cast<uint64_t>(v);
    }

    const picojson::array& json_array(const picojson::object& obj, const std::string& key) {
        const picojson::value& v = require_field(obj, key);
        if (!v.is<picojson::array>()) throw std::runtime_error("JSON field must be array: " + key);
        return v.get<picojson::array>();
    }

    uint8_t checked_u8(int64_t v, const std::string& key) {
        if (v < 0 || v > 255) throw std::runtime_error("field does not fit uint8: " + key);
        return static_cast<uint8_t>(v);
    }
} // namespace

Meta load_meta(const std::filesystem::path& output_dir) {
    std::ifstream in(output_dir / "meta.json");
    if (!in) throw std::runtime_error("missing meta.json: " + (output_dir / "meta.json").string());
    picojson::value root;
    std::string err = picojson::parse(root, in);
    if (!err.empty()) throw std::runtime_error("failed to parse meta.json: " + err);
    const auto& obj = require_object(root, "meta.json");

    Meta m;
    if (obj.count("index_file")) m.index_file = json_string(obj, "index_file");
    m.index_record_count = obj.count("index_record_count") ? json_u64(obj, "index_record_count") : 0;
    m.group_size = json_u64(obj, "group_size_N");
    m.file_name_bytes = json_u64(obj, "file_name_bytes");
    m.binary_index_record_size = json_u64(obj, "binary_index_record_size");
    if (m.group_size != kGroupSize) throw std::runtime_error(
        "meta group_size_N does not match compile-time kGroupSize");
    if (m.file_name_bytes != kFileNameBytes) throw std::runtime_error(
        "meta file_name_bytes does not match compile-time kFileNameBytes");
    size_t expected = binary_index_record_size(m.group_size, m.file_name_bytes);
    if (m.binary_index_record_size != expected) {
        std::ostringstream oss;
        oss << "binary_index_record_size=" << m.binary_index_record_size << ", expected=" << expected;
        throw std::runtime_error(oss.str());
    }
    return m;
}

IndexRecordFixed parse_index_line(const std::string& line, size_t line_no, const Meta& meta) {
    picojson::value root;
    std::string err = picojson::parse(root, line);
    if (!err.empty()) throw std::runtime_error(
        "failed to parse index JSONL line " + std::to_string(line_no) + ": " + err);
    const auto& obj = require_object(root, "index JSONL line");

    IndexRecordFixed rec{};
    rec.item_no = json_u64(obj, "item_no");
    rec.seq = checked_u8(json_i64(obj, "seq"), "seq");
    rec.dtype = checked_u8(json_i64(obj, "dtype_id"), "dtype_id");
    rec.avg_timestamp = json_double(obj, "avg_timestamp");
    rec.avg_timestamp_us = json_i64(obj, "avg_timestamp_us");
    rec.slot_count = checked_u8(json_i64(obj, "slot_count"), "slot_count");
    rec.valid_count = checked_u8(json_i64(obj, "valid_count"), "valid_count");
    if (rec.slot_count != meta.group_size) throw std::runtime_error(
        "slot_count does not match meta at line " + std::to_string(line_no));
    if (rec.valid_count > rec.slot_count) throw std::runtime_error(
        "valid_count > slot_count at line " + std::to_string(line_no));
    rec.index_copy_offset = json_u64(obj, "index_copy_offset");
    rec.binary_index_record_size = json_u64(obj, "binary_index_record_size");
    if (rec.binary_index_record_size != meta.binary_index_record_size) throw std::runtime_error(
        "binary_index_record_size mismatch at line " + std::to_string(line_no));
    set_fixed_file_name(rec, json_string(obj, "file_name"), meta.file_name_bytes);

    const auto& vec = json_array(obj, "index_vector");
    if (vec.size() != 4) throw std::runtime_error(
        "index_vector length must be 4 at line " + std::to_string(line_no));
    for (size_t i = 0; i < 4; ++i) rec.index_vector[i] = json_double_value(vec[i], "index_vector");

    const auto& ts = json_array(obj, "timestamps");
    const auto& ts_us = json_array(obj, "timestamp_us");
    const auto& offsets = json_array(obj, "data_offsets");
    const auto& sizes = json_array(obj, "sizes");
    const size_t n = meta.group_size;
    if (ts.size() != n || ts_us.size() != n || sizes.size() != n || offsets.size() != n + 1) {
        throw std::runtime_error("fixed array length mismatch at line " + std::to_string(line_no));
    }
    for (size_t i = 0; i < n; ++i) {
        rec.timestamps[i] = json_double_value(ts[i], "timestamps");
        rec.timestamp_us[i] = json_i64_value(ts_us[i], "timestamp_us");
        int64_t sz = json_i64_value(sizes[i], "sizes");
        if (sz < 0) throw std::runtime_error("negative size at line " + std::to_string(line_no));
        rec.sizes[i] = static_cast<uint64_t>(sz);
    }
    for (size_t i = 0; i < n + 1; ++i) {
        int64_t off = json_i64_value(offsets[i], "data_offsets");
        if (off < 0) throw std::runtime_error("negative data offset at line " + std::to_string(line_no));
        rec.data_offsets[i] = static_cast<uint64_t>(off);
    }

    if (rec.data_offsets[0] != 0) throw std::runtime_error(
        "data_offsets[0] must be 0 at line " + std::to_string(line_no));
    for (size_t i = 0; i < n; ++i) {
        if (rec.data_offsets[i + 1] < rec.data_offsets[i]) throw std::runtime_error(
            "data_offsets not monotonic at line " + std::to_string(line_no));
        const uint64_t derived_size = rec.data_offsets[i + 1] - rec.data_offsets[i];
        if (derived_size != rec.sizes[i]) throw std::runtime_error(
            "sizes[i] != data_offsets[i+1]-data_offsets[i] at line " + std::to_string(line_no));
        if (i >= rec.valid_count && rec.sizes[i] != 0) throw std::runtime_error(
            "padding slot has non-zero size at line " + std::to_string(line_no));
    }
    return rec;
}

std::vector<IndexRecordFixed> load_index_records(const std::filesystem::path& output_dir, const Meta& meta) {
    const std::filesystem::path index_path = output_dir / meta.index_file;
    std::ifstream in(index_path);
    if (!in) throw std::runtime_error("missing index file: " + index_path.string());
    std::vector<IndexRecordFixed> records;
    records.reserve(meta.index_record_count);
    std::string line;
    size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (trim(line).empty()) continue;
        records.push_back(parse_index_line(line, line_no, meta));
    }
    if (meta.index_record_count != 0 && records.size() != meta.index_record_count) {
        throw std::runtime_error("index_record_count does not match actual JSONL line count");
    }
    return records;
}

std::vector<IndexRecordFixed> load_candidate_records(const std::filesystem::path& output_dir,
                                                      const Meta& meta,
                                                      const std::vector<size_t>& candidates) {
    if (candidates.empty()) return {};
    const std::filesystem::path index_path = output_dir / meta.index_file;
    std::ifstream in(index_path);
    if (!in) throw std::runtime_error("missing index file: " + index_path.string());

    std::unordered_set<uint64_t> want;
    want.reserve(candidates.size() * 2 + 1);
    for (size_t idx : candidates) want.insert(static_cast<uint64_t>(idx));

    std::vector<IndexRecordFixed> records;
    records.reserve(want.size());
    std::string line;
    size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (trim(line).empty()) continue;
        IndexRecordFixed rec = parse_index_line(line, line_no, meta);
        if (want.erase(rec.item_no) > 0) {
            records.push_back(std::move(rec));
            if (want.empty()) break;
        }
    }

    if (!want.empty()) {
        std::ostringstream oss;
        oss << "missing " << want.size() << " candidate index record(s)";
        size_t shown = 0;
        oss << " (example item_no: ";
        for (uint64_t v : want) {
            oss << v;
            if (++shown >= 3 || shown == want.size()) break;
            oss << ", ";
        }
        oss << ")";
        throw std::runtime_error(oss.str());
    }

    std::sort(records.begin(), records.end(), [](const IndexRecordFixed& a, const IndexRecordFixed& b) {
        return a.item_no < b.item_no;
    });
    return records;
}

} // namespace file_index
} // namespace tblaeq
