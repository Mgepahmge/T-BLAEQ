// IndexJsonTblaeqTool.cpp
// Build JSONL index + binary payloads and co-located T-BLAEQ index, then query both.

#include "CLI11.hpp"
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wparentheses"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wdangling-reference"
#endif
#define PICOJSON_USE_INT64
#include "picojson.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "src/core/MemoryPolicy.cuh"
#include "src/core/QueryHandler.cuh"

namespace fs = std::filesystem;

namespace {

constexpr size_t MAX_N = 255;
constexpr size_t MAX_FILE_NAME_BYTES = 256;
constexpr size_t kIndexVectorDim = 4;

#ifndef TBLAEQ_INDEX_GROUP_SIZE
constexpr size_t kGroupSize = 1;
#else
constexpr size_t kGroupSize = TBLAEQ_INDEX_GROUP_SIZE;
#endif

#ifndef TBLAEQ_FILE_NAME_BYTES
constexpr size_t kFileNameBytes = 32;
#else
constexpr size_t kFileNameBytes = TBLAEQ_FILE_NAME_BYTES;
#endif

static_assert(kGroupSize >= 1 && kGroupSize <= MAX_N, "kGroupSize must be in [1, 255]");
static_assert(kFileNameBytes >= 1 && kFileNameBytes <= MAX_FILE_NAME_BYTES,
              "kFileNameBytes must be in [1, 256]");

constexpr uint8_t TYPE_IMAGE = 0;
constexpr uint8_t TYPE_LIDAR_360 = 1;
constexpr uint8_t TYPE_LIVOX_AVIA = 2;
constexpr uint8_t TYPE_RADAR_ENCHANCE_PCL = 3;
constexpr uint8_t TYPE_COORD = 4;

constexpr const char* kDefaultIndexDir = "file_index";
constexpr const char* kDefaultTblaeqDir = "tblaeq_index";

struct TypeInfo {
    const char* folder_name;
    const char* dtype_name;
    uint8_t dtype;
    const char* extension;
};

const std::vector<TypeInfo> kFolderTypes = {
    {"Image", "image", TYPE_IMAGE, ".png"},
    {"lidar_360", "lidar_360", TYPE_LIDAR_360, ".npy"},
    {"livox_avia", "livox_avia", TYPE_LIVOX_AVIA, ".npy"},
    {"radar_enchance_pcl", "radar_enchance_pcl", TYPE_RADAR_ENCHANCE_PCL, ".npy"},
};

struct BuildArgs {
    std::string root;
    std::string csv_path;
    std::string index_root;
    std::string index_dir = kDefaultIndexDir;
    std::string tblaeq_dir = kDefaultTblaeqDir;
    std::string index_file = "index.jsonl";
    std::string max_data_file_size_text = "1GiB";
    uint64_t max_data_file_size = 1024ULL * 1024ULL * 1024ULL;
    bool verify = false;
    bool force_cpu = false;
    size_t height = IndexBuilder::kDefaultHeight;
    std::string ratios_text = "100,50,20";
};

struct QueryArgs {
    std::string index_root;
    std::string index_dir = kDefaultIndexDir;
    std::string tblaeq_dir = kDefaultTblaeqDir;
    std::string mode = "knn"; // knn | range
    double timestamp = 0.0;
    bool has_timestamp = false;
    double start = 0.0;
    bool has_start = false;
    double end = 0.0;
    bool has_end = false;
    uint8_t seq = 0;
    bool has_seq = false;
    size_t knn_k = 10;
    size_t range_limit = 0; // 0 means all
    bool include_padding = false;
    std::string dump_dir;
    bool has_dump_dir = false;
};

struct SourceRecord {
    std::string seq_name;
    uint8_t seq = 0;
    std::string dtype_name;
    uint8_t dtype = 0;
    double timestamp = 0.0;
    std::optional<fs::path> source_path;
    std::vector<uint8_t> payload;

    uint64_t size() const {
        if (!payload.empty()) return static_cast<uint64_t>(payload.size());
        if (!source_path.has_value()) throw std::runtime_error("record has no payload or source path");
        return static_cast<uint64_t>(fs::file_size(*source_path));
    }
};

struct IndexGroup {
    uint8_t seq = 0;
    std::string seq_name;
    std::string dtype_name;
    uint8_t dtype = 0;
    std::vector<SourceRecord> records;
    double avg_ts = 0.0;
};

struct IndexRecordFixed {
    uint64_t item_no = 0;
    double index_vector[4]{};
    uint8_t slot_count = 0;
    uint8_t dtype = 0;
    uint8_t seq = 0;
    uint8_t valid_count = 0;
    double avg_timestamp = 0.0;
    int64_t avg_timestamp_us = 0;
    double timestamps[MAX_N]{};
    int64_t timestamp_us[MAX_N]{};
    char file_name[MAX_FILE_NAME_BYTES]{};
    uint64_t index_copy_offset = 0;
    uint64_t binary_index_record_size = 0;
    uint64_t data_offsets[MAX_N + 1]{};
    uint64_t sizes[MAX_N]{};
};

struct Meta {
    std::string index_file = "index.jsonl";
    size_t index_record_count = 0;
    size_t group_size = 0;
    size_t file_name_bytes = 0;
    size_t binary_index_record_size = 0;
};

struct SlotRef {
    size_t record_idx = 0;
    uint8_t slot = 0;
    int64_t timestamp_us = 0;
    double timestamp = 0.0;
    uint64_t size = 0;
};

struct Match {
    SlotRef slot_ref;
    int64_t diff_us = 0;
};

bool host_is_little_endian() {
    uint16_t x = 1;
    return *reinterpret_cast<uint8_t*>(&x) == 1;
}

uint64_t bswap64(uint64_t x) {
    return ((x & 0x00000000000000FFULL) << 56) |
           ((x & 0x000000000000FF00ULL) << 40) |
           ((x & 0x0000000000FF0000ULL) << 24) |
           ((x & 0x00000000FF000000ULL) << 8)  |
           ((x & 0x000000FF00000000ULL) >> 8)  |
           ((x & 0x0000FF0000000000ULL) >> 24) |
           ((x & 0x00FF000000000000ULL) >> 40) |
           ((x & 0xFF00000000000000ULL) >> 56);
}

void append_u8(std::vector<uint8_t>& out, uint8_t v) { out.push_back(v); }

void append_u64_le(std::vector<uint8_t>& out, uint64_t v) {
    if (!host_is_little_endian()) v = bswap64(v);
    uint8_t b[8];
    std::memcpy(b, &v, 8);
    out.insert(out.end(), b, b + 8);
}

void append_double_le(std::vector<uint8_t>& out, double v) {
    static_assert(sizeof(double) == 8, "double must be 8 bytes");
    uint64_t u = 0;
    std::memcpy(&u, &v, 8);
    append_u64_le(out, u);
}

std::string trim(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

std::string to_lower_ascii(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string strip_utf8_bom(std::string s) {
    if (s.size() >= 3 &&
        static_cast<unsigned char>(s[0]) == 0xEF &&
        static_cast<unsigned char>(s[1]) == 0xBB &&
        static_cast<unsigned char>(s[2]) == 0xBF) {
        return s.substr(3);
    }
    return s;
}

uint64_t parse_size_arg(const std::string& text) {
    const std::string s = to_lower_ascii(trim(text));
    static const std::regex re(R"(^(\d+)([a-z]+)?$)");
    std::smatch m;
    if (!std::regex_match(s, m, re)) throw std::runtime_error("invalid size: " + text);
    uint64_t value = std::stoull(m[1].str());
    const std::string unit = m[2].matched ? m[2].str() : "b";
    uint64_t mul = 1;
    if (unit == "b") mul = 1;
    else if (unit == "kb") mul = 1000ULL;
    else if (unit == "kib") mul = 1024ULL;
    else if (unit == "mb") mul = 1000ULL * 1000ULL;
    else if (unit == "mib") mul = 1024ULL * 1024ULL;
    else if (unit == "gb") mul = 1000ULL * 1000ULL * 1000ULL;
    else if (unit == "gib") mul = 1024ULL * 1024ULL * 1024ULL;
    else throw std::runtime_error("invalid size unit: " + text);
    if (value > std::numeric_limits<uint64_t>::max() / mul) throw std::runtime_error("size overflow: " + text);
    return value * mul;
}

uint8_t parse_seq(const std::string& seq_name) {
    static const std::regex re(R"(seq(\d+))");
    std::smatch m;
    if (!std::regex_match(seq_name, m, re)) throw std::runtime_error("invalid sequence name: " + seq_name);
    unsigned long v = std::stoul(m[1].str());
    if (v > 255UL) throw std::runtime_error("seq does not fit uint8: " + seq_name);
    return static_cast<uint8_t>(v);
}

uint8_t parse_seq_arg(const std::string& s0) {
    std::string s = trim(s0);
    if (s.rfind("seq", 0) == 0) s = s.substr(3);
    unsigned long v = std::stoul(s);
    if (v > 255UL) throw std::runtime_error("seq does not fit uint8: " + s0);
    return static_cast<uint8_t>(v);
}

double parse_timestamp_from_filename(const fs::path& p) {
    const std::string stem = p.stem().string();
    size_t pos = 0;
    double v = std::stod(stem, &pos);
    if (pos != stem.size()) throw std::runtime_error("filename stem is not a pure timestamp: " + p.string());
    return v;
}

int64_t timestamp_to_us(double timestamp) {
    const long double scaled = static_cast<long double>(timestamp) * 1000000.0L;
    if (scaled < static_cast<long double>(std::numeric_limits<int64_t>::min()) ||
        scaled > static_cast<long double>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error("timestamp microseconds overflow int64");
    }
    return static_cast<int64_t>(std::llround(scaled));
}

std::array<double, 3> split_timestamp_3d(double timestamp) {
    const int64_t signed_us = timestamp_to_us(timestamp);
    if (signed_us < 0) throw std::runtime_error("negative timestamp is not supported");
    const uint64_t ts_us = static_cast<uint64_t>(signed_us);
    constexpr uint64_t mask = (1ULL << 21) - 1ULL;
    return {
        static_cast<double>(((ts_us >> 42) & mask) + 1ULL),
        static_cast<double>(((ts_us >> 21) & mask) + 1ULL),
        static_cast<double>((ts_us & mask) + 1ULL)
    };
}

size_t binary_index_record_size(size_t n, size_t file_name_bytes) {
    return 4 * 8 + 1 + 1 + n * 8 + 1 + file_name_bytes + 8 + (n + 1) * 8;
}

std::string binary_struct_format_string(size_t n, size_t file_name_bytes) {
    std::ostringstream oss;
    oss << "<4dBB" << n << "dB" << file_name_bytes << "sQ" << (n + 1) << "Q";
    return oss.str();
}

std::string fixed_file_name_to_string(const char file_name[MAX_FILE_NAME_BYTES]) {
    size_t n = 0;
    while (n < MAX_FILE_NAME_BYTES && file_name[n] != '\0') ++n;
    return std::string(file_name, n);
}

void set_fixed_file_name(IndexRecordFixed& rec, const std::string& name, size_t file_name_bytes) {
    if (file_name_bytes > MAX_FILE_NAME_BYTES) throw std::runtime_error("file_name_bytes exceeds MAX_FILE_NAME_BYTES");
    if (name.size() > file_name_bytes) throw std::runtime_error("data file name exceeds fixed file_name_bytes: " + name);
    std::memset(rec.file_name, 0, sizeof(rec.file_name));
    std::memcpy(rec.file_name, name.data(), name.size());
}

std::vector<uint8_t> encode_binary_index_copy(const IndexRecordFixed& rec, size_t n, size_t file_name_bytes) {
    if (n == 0 || n > MAX_N) throw std::runtime_error("invalid N while encoding binary index");
    if (file_name_bytes == 0 || file_name_bytes > MAX_FILE_NAME_BYTES) throw std::runtime_error("invalid file_name_bytes");
    std::vector<uint8_t> out;
    out.reserve(binary_index_record_size(n, file_name_bytes));
    for (size_t i = 0; i < 4; ++i) append_double_le(out, rec.index_vector[i]);
    append_u8(out, rec.slot_count);
    append_u8(out, rec.dtype);
    for (size_t i = 0; i < n; ++i) append_double_le(out, rec.timestamps[i]);
    append_u8(out, rec.seq);
    out.insert(out.end(), reinterpret_cast<const uint8_t*>(rec.file_name),
               reinterpret_cast<const uint8_t*>(rec.file_name) + file_name_bytes);
    append_u64_le(out, rec.index_copy_offset);
    for (size_t i = 0; i < n + 1; ++i) append_u64_le(out, rec.data_offsets[i]);
    if (out.size() != binary_index_record_size(n, file_name_bytes)) throw std::runtime_error("binary index size mismatch");
    return out;
}

std::string extension_lower(const fs::path& p) { return to_lower_ascii(p.extension().string()); }

std::vector<fs::directory_entry> sorted_entries(const fs::path& dir) {
    std::vector<fs::directory_entry> entries;
    if (!fs::exists(dir)) return entries;
    for (const auto& e : fs::directory_iterator(dir)) entries.push_back(e);
    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
        return a.path().filename().string() < b.path().filename().string();
    });
    return entries;
}

std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string cur;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') { cur.push_back('"'); ++i; }
                else in_quotes = false;
            } else cur.push_back(c);
        } else {
            if (c == '"') in_quotes = true;
            else if (c == ',') { fields.push_back(cur); cur.clear(); }
            else cur.push_back(c);
        }
    }
    fields.push_back(cur);
    return fields;
}

std::array<double, 3> parse_position(const std::string& value) {
    std::string s = trim(value);
    if (s.size() < 2 || s.front() != '[' || s.back() != ']') throw std::runtime_error("bad Position: " + value);
    s = s.substr(1, s.size() - 2);
    std::vector<double> vals;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, ',')) vals.push_back(std::stod(trim(part)));
    if (vals.size() != 3) throw std::runtime_error("Position must have 3 numbers: " + value);
    return {vals[0], vals[1], vals[2]};
}

std::vector<uint8_t> encode_coord_payload(double x, double y, double z, uint8_t cls) {
    std::vector<uint8_t> out;
    out.reserve(32);
    append_double_le(out, x);
    append_double_le(out, y);
    append_double_le(out, z);
    append_u8(out, cls);
    for (int i = 0; i < 7; ++i) append_u8(out, 0);
    return out;
}

std::vector<SourceRecord> collect_file_records(const fs::path& root) {
    std::vector<SourceRecord> records;
    static const std::regex seq_re(R"(seq\d+)");
    for (const auto& seq_entry : sorted_entries(root)) {
        if (!seq_entry.is_directory()) continue;
        const std::string seq_name = seq_entry.path().filename().string();
        if (!std::regex_match(seq_name, seq_re)) continue;
        uint8_t seq = parse_seq(seq_name);
        for (const auto& ti : kFolderTypes) {
            const fs::path folder = seq_entry.path() / ti.folder_name;
            if (!fs::exists(folder)) continue;
            if (!fs::is_directory(folder)) throw std::runtime_error("expected directory: " + folder.string());
            for (const auto& data_entry : sorted_entries(folder)) {
                if (!data_entry.is_regular_file()) continue;
                if (extension_lower(data_entry.path()) != ti.extension) continue;
                SourceRecord rec;
                rec.seq_name = seq_name;
                rec.seq = seq;
                rec.dtype_name = ti.dtype_name;
                rec.dtype = ti.dtype;
                rec.timestamp = parse_timestamp_from_filename(data_entry.path());
                rec.source_path = data_entry.path();
                records.push_back(std::move(rec));
            }
        }
    }
    return records;
}

void collect_coord_records(const fs::path& csv_path, std::vector<SourceRecord>& records) {
    std::ifstream in(csv_path);
    if (!in) throw std::runtime_error("failed to open CSV: " + csv_path.string());
    std::string header;
    if (!std::getline(in, header)) throw std::runtime_error("empty CSV: " + csv_path.string());
    if (!header.empty() && header.back() == '\r') header.pop_back();
    auto headers = parse_csv_line(header);
    if (!headers.empty()) headers[0] = strip_utf8_bom(headers[0]);
    std::map<std::string, size_t> col;
    for (size_t i = 0; i < headers.size(); ++i) col[trim(headers[i])] = i;
    for (const char* name : {"Sequence", "Timestamp", "Position", "Classification"}) {
        if (!col.count(name)) throw std::runtime_error(std::string("CSV missing column: ") + name);
    }
    std::string line;
    size_t row_no = 1;
    while (std::getline(in, line)) {
        ++row_no;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (trim(line).empty()) continue;
        auto fields = parse_csv_line(line);
        auto get = [&](const std::string& name) -> std::string {
            size_t i = col.at(name);
            if (i >= fields.size()) throw std::runtime_error("CSV row missing field at row " + std::to_string(row_no));
            return trim(fields[i]);
        };
        SourceRecord rec;
        rec.seq_name = get("Sequence");
        rec.seq = parse_seq(rec.seq_name);
        rec.dtype_name = "coord";
        rec.dtype = TYPE_COORD;
        rec.timestamp = std::stod(get("Timestamp"));
        auto pos = parse_position(get("Position"));
        int cls = std::stoi(get("Classification"));
        if (cls < 0 || cls > 255) throw std::runtime_error("Classification does not fit uint8 at row " + std::to_string(row_no));
        rec.payload = encode_coord_payload(pos[0], pos[1], pos[2], static_cast<uint8_t>(cls));
        records.push_back(std::move(rec));
    }
}

std::vector<IndexGroup> make_groups(std::vector<SourceRecord> records, size_t group_size) {
    std::map<std::pair<int, int>, std::vector<SourceRecord>> buckets;
    for (auto& rec : records) buckets[{static_cast<int>(rec.seq), static_cast<int>(rec.dtype)}].push_back(std::move(rec));
    std::vector<IndexGroup> groups;
    for (auto& kv : buckets) {
        auto& bucket = kv.second;
        std::sort(bucket.begin(), bucket.end(), [](const auto& a, const auto& b) { return a.timestamp < b.timestamp; });
        for (size_t start = 0; start < bucket.size(); start += group_size) {
            size_t end = std::min(bucket.size(), start + group_size);
            IndexGroup g;
            g.records.reserve(end - start);
            double sum = 0.0;
            for (size_t i = start; i < end; ++i) { sum += bucket[i].timestamp; g.records.push_back(std::move(bucket[i])); }
            g.seq = g.records.front().seq;
            g.seq_name = g.records.front().seq_name;
            g.dtype_name = g.records.front().dtype_name;
            g.dtype = g.records.front().dtype;
            g.avg_ts = sum / static_cast<double>(g.records.size());
            groups.push_back(std::move(g));
        }
    }
    std::stable_sort(groups.begin(), groups.end(), [](const auto& a, const auto& b) {
        if (a.seq != b.seq) return a.seq < b.seq;
        return a.avg_ts < b.avg_ts;
    });
    return groups;
}

picojson::value jnum_i64(int64_t v) { return picojson::value(v); }
picojson::value jnum_u64(uint64_t v) {
    if (v > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) throw std::runtime_error("uint64 too large for JSON int64");
    return picojson::value(static_cast<int64_t>(v));
}

picojson::array json_double_array(const double* values, size_t n) {
    picojson::array a;
    a.reserve(n);
    for (size_t i = 0; i < n; ++i) a.push_back(picojson::value(values[i]));
    return a;
}

picojson::array json_i64_array(const int64_t* values, size_t n) {
    picojson::array a;
    a.reserve(n);
    for (size_t i = 0; i < n; ++i) a.push_back(jnum_i64(values[i]));
    return a;
}

picojson::array json_u64_array(const uint64_t* values, size_t n) {
    picojson::array a;
    a.reserve(n);
    for (size_t i = 0; i < n; ++i) a.push_back(jnum_u64(values[i]));
    return a;
}

picojson::value index_record_to_json(const IndexRecordFixed& rec, const std::string& dtype_name, const std::string& seq_name) {
    const size_t n = rec.slot_count;
    picojson::object o;
    o["item_no"] = jnum_u64(rec.item_no);
    o["seq"] = jnum_i64(rec.seq);
    o["seq_name"] = picojson::value(seq_name);
    o["dtype"] = picojson::value(dtype_name);
    o["dtype_id"] = jnum_i64(rec.dtype);
    o["avg_timestamp"] = picojson::value(rec.avg_timestamp);
    o["avg_timestamp_us"] = jnum_i64(rec.avg_timestamp_us);
    o["index_vector"] = picojson::value(json_double_array(rec.index_vector, 4));
    o["timestamps"] = picojson::value(json_double_array(rec.timestamps, n));
    o["timestamp_us"] = picojson::value(json_i64_array(rec.timestamp_us, n));
    o["slot_count"] = jnum_i64(rec.slot_count);
    o["valid_count"] = jnum_i64(rec.valid_count);
    o["file_name"] = picojson::value(fixed_file_name_to_string(rec.file_name));
    o["index_copy_offset"] = jnum_u64(rec.index_copy_offset);
    o["binary_index_record_size"] = jnum_u64(rec.binary_index_record_size);
    o["data_base_offset"] = jnum_u64(rec.index_copy_offset + rec.binary_index_record_size);
    o["data_offsets"] = picojson::value(json_u64_array(rec.data_offsets, n + 1));
    o["sizes"] = picojson::value(json_u64_array(rec.sizes, n));
    return picojson::value(o);
}

IndexRecordFixed build_fixed_index_record(const IndexGroup& group, size_t item_no, size_t group_size,
                                          size_t file_name_bytes, const std::string& data_file_name,
                                          uint64_t index_copy_offset) {
    if (group.records.empty()) throw std::runtime_error("empty group");
    IndexRecordFixed rec{};
    rec.item_no = static_cast<uint64_t>(item_no);
    rec.slot_count = static_cast<uint8_t>(group_size);
    rec.dtype = group.dtype;
    rec.seq = group.seq;
    rec.valid_count = static_cast<uint8_t>(group.records.size());
    rec.avg_timestamp = group.avg_ts;
    rec.avg_timestamp_us = timestamp_to_us(group.avg_ts);
    rec.index_copy_offset = index_copy_offset;
    rec.binary_index_record_size = binary_index_record_size(group_size, file_name_bytes);
    set_fixed_file_name(rec, data_file_name, file_name_bytes);

    auto v3 = split_timestamp_3d(group.avg_ts);
    rec.index_vector[0] = v3[0];
    rec.index_vector[1] = v3[1];
    rec.index_vector[2] = v3[2];
    rec.index_vector[3] = static_cast<double>(group.seq);

    std::vector<const SourceRecord*> sorted;
    sorted.reserve(group.records.size());
    for (const auto& r : group.records) sorted.push_back(&r);
    std::sort(sorted.begin(), sorted.end(), [](const auto* a, const auto* b) { return a->timestamp < b->timestamp; });

    for (size_t i = 0; i < sorted.size(); ++i) {
        rec.timestamps[i] = sorted[i]->timestamp;
        rec.timestamp_us[i] = timestamp_to_us(sorted[i]->timestamp);
        rec.sizes[i] = sorted[i]->size();
    }
    const double last_ts = rec.timestamps[sorted.size() - 1];
    const int64_t last_ts_us = rec.timestamp_us[sorted.size() - 1];
    for (size_t i = sorted.size(); i < group_size; ++i) {
        rec.timestamps[i] = last_ts;
        rec.timestamp_us[i] = last_ts_us;
        rec.sizes[i] = 0;
    }
    uint64_t total = 0;
    rec.data_offsets[0] = 0;
    for (size_t i = 0; i < group_size; ++i) {
        if (total > std::numeric_limits<uint64_t>::max() - rec.sizes[i]) throw std::runtime_error("data offset overflow");
        total += rec.sizes[i];
        rec.data_offsets[i + 1] = total;
    }
    return rec;
}

void copy_file_bytes(const fs::path& src, std::ostream& out) {
    std::ifstream in(src, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open input data: " + src.string());
    std::vector<char> buf(1024 * 1024);
    while (in) {
        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        std::streamsize n = in.gcount();
        if (n > 0) {
            out.write(buf.data(), n);
            if (!out) throw std::runtime_error("failed writing payload");
        }
    }
}

class DataFileWriter {
public:
    DataFileWriter(fs::path output_dir, uint64_t max_size, size_t file_name_bytes)
        : output_dir_(std::move(output_dir)), max_size_(max_size), file_name_bytes_(file_name_bytes) {}

    std::pair<std::string, uint64_t> reserve_group(uint64_t group_total_size) {
        if (group_total_size > max_size_) throw std::runtime_error("one index group exceeds max data file size");
        if (!out_.is_open()) open_next();
        if (current_offset_ > 0 && current_offset_ + group_total_size > max_size_) open_next();
        return {current_name_, current_offset_};
    }

    std::ostream& stream() { return out_; }
    void advance(uint64_t n) { current_offset_ += n; }
    void close() { if (out_.is_open()) out_.close(); }
    const std::vector<std::string>& created_names() const { return created_names_; }

private:
    void open_next() {
        if (out_.is_open()) out_.close();
        std::ostringstream name;
        name << "data_" << std::setw(4) << std::setfill('0') << created_names_.size() << ".bin";
        current_name_ = name.str();
        if (current_name_.size() > file_name_bytes_) {
            throw std::runtime_error("data file name exceeds fixed file_name_bytes: " + current_name_);
        }
        fs::path path = output_dir_ / current_name_;
        out_.open(path, std::ios::binary);
        if (!out_) throw std::runtime_error("failed to open data file: " + path.string());
        created_names_.push_back(current_name_);
        current_offset_ = 0;
    }

    fs::path output_dir_;
    uint64_t max_size_ = 0;
    size_t file_name_bytes_ = 0;
    std::ofstream out_;
    std::string current_name_;
    uint64_t current_offset_ = 0;
    std::vector<std::string> created_names_;
};

void verify_binary_copies(const fs::path& output_dir, const std::vector<IndexRecordFixed>& records) {
    for (const auto& rec : records) {
        const fs::path data_path = output_dir / fixed_file_name_to_string(rec.file_name);
        if (!fs::exists(data_path)) throw std::runtime_error("data file not found: " + data_path.string());
        uint64_t file_size = static_cast<uint64_t>(fs::file_size(data_path));
        if (rec.index_copy_offset > file_size || rec.binary_index_record_size > file_size - rec.index_copy_offset) {
            throw std::runtime_error("binary index copy exceeds data file length: " + data_path.string());
        }
        std::ifstream in(data_path, std::ios::binary);
        if (!in) throw std::runtime_error("failed to open data file: " + data_path.string());
        std::vector<uint8_t> expected = encode_binary_index_copy(rec, kGroupSize, kFileNameBytes);
        std::vector<uint8_t> actual(expected.size());
        in.seekg(static_cast<std::streamoff>(rec.index_copy_offset), std::ios::beg);
        in.read(reinterpret_cast<char*>(actual.data()), static_cast<std::streamsize>(actual.size()));
        if (in.gcount() != static_cast<std::streamsize>(actual.size())) throw std::runtime_error("failed to read binary index copy");
        if (actual != expected) throw std::runtime_error("binary index copy mismatch at item " + std::to_string(rec.item_no));
    }
}

void write_meta(const fs::path& output_dir, const BuildArgs& args, size_t record_count, const std::vector<std::string>& data_files) {
    picojson::object meta;
    meta["format"] = picojson::value("index-jsonl-data-binary-v1");
    meta["main_index_format"] = picojson::value("jsonl");
    meta["data_index_copy_format"] = picojson::value("fixed_binary");
    meta["endianness"] = picojson::value("little");
    meta["index_file"] = picojson::value(args.index_file);
    meta["index_record_count"] = jnum_u64(record_count);
    meta["group_size_N"] = jnum_i64(static_cast<int64_t>(kGroupSize));
    meta["file_name_bytes"] = jnum_u64(kFileNameBytes);
    meta["binary_index_record_size"] = jnum_u64(binary_index_record_size(kGroupSize, kFileNameBytes));
    meta["binary_index_struct_format"] = picojson::value(binary_struct_format_string(kGroupSize, kFileNameBytes));
    meta["max_runtime_N"] = jnum_u64(MAX_N);
    meta["max_runtime_file_name_bytes"] = jnum_u64(MAX_FILE_NAME_BYTES);

    picojson::object type_enum;
    type_enum["image"] = jnum_i64(TYPE_IMAGE);
    type_enum["lidar_360"] = jnum_i64(TYPE_LIDAR_360);
    type_enum["livox_avia"] = jnum_i64(TYPE_LIVOX_AVIA);
    type_enum["radar_enchance_pcl"] = jnum_i64(TYPE_RADAR_ENCHANCE_PCL);
    type_enum["coord"] = jnum_i64(TYPE_COORD);
    meta["type_enum"] = picojson::value(type_enum);

    picojson::object coord;
    coord["format"] = picojson::value("<dddB7x");
    coord["size"] = jnum_i64(32);
    coord["fields"] = picojson::value(picojson::array{
        picojson::value("x_float64"), picojson::value("y_float64"), picojson::value("z_float64"),
        picojson::value("classification_uint8"), picojson::value("padding_7_bytes")
    });
    meta["coordinate_payload"] = picojson::value(coord);

    meta["timestamp_vector_rule"] = picojson::value(
        "round(avg_timestamp * 1_000_000) to integer microseconds; split into three 21-bit limbs [bits 42..62, 21..41, 0..20]; store each limb + 1 as float64");
    meta["data_access_rule"] = picojson::value("data_base = index_copy_offset + binary_index_record_size; payload_start = data_base + data_offsets[i]; payload_end = data_base + data_offsets[i+1]");
    meta["max_data_file_size"] = jnum_u64(args.max_data_file_size);

    picojson::array df;
    for (const auto& name : data_files) df.push_back(picojson::value(name));
    meta["data_files"] = picojson::value(df);

    std::ofstream out(output_dir / "meta.json");
    if (!out) throw std::runtime_error("failed to write meta.json");
    out << picojson::value(meta).serialize(true);
}

std::vector<size_t> parse_ratios_csv(const std::string& s) {
    std::vector<size_t> ratios;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) throw std::invalid_argument("Invalid --ratios: empty segment in '" + s + "'");
        const size_t r = static_cast<size_t>(std::stoull(token));
        if (r == 0) throw std::invalid_argument("Invalid --ratios: ratio must be > 0");
        ratios.push_back(r);
    }
    if (ratios.empty()) throw std::invalid_argument("Invalid --ratios: no ratios parsed from '" + s + "'");
    return ratios;
}

int run_build(BuildArgs args) {
    fs::path root(args.root);
    fs::path index_root(args.index_root);
    if (!fs::exists(root) || !fs::is_directory(root)) throw std::runtime_error("root is not a directory: " + root.string());
    fs::create_directories(index_root);
    fs::path index_dir = index_root / args.index_dir;
    fs::path tblaeq_dir = index_root / args.tblaeq_dir;
    fs::create_directories(index_dir);
    fs::create_directories(tblaeq_dir);

    std::vector<SourceRecord> records = collect_file_records(root);
    collect_coord_records(fs::path(args.csv_path), records);
    if (records.empty()) throw std::runtime_error("no input records found");

    const size_t input_record_count = records.size();
    auto groups = make_groups(std::move(records), kGroupSize);
    const size_t bin_rec_size = binary_index_record_size(kGroupSize, kFileNameBytes);

    std::ofstream index_out(index_dir / args.index_file);
    if (!index_out) throw std::runtime_error("failed to write main index: " + (index_dir / args.index_file).string());

    DataFileWriter data_writer(index_dir, args.max_data_file_size, kFileNameBytes);
    std::vector<IndexRecordFixed> fixed_records;
    fixed_records.reserve(groups.size());

    for (size_t item_no = 0; item_no < groups.size(); ++item_no) {
        auto& group = groups[item_no];
        std::sort(group.records.begin(), group.records.end(), [](const auto& a, const auto& b) { return a.timestamp < b.timestamp; });
        uint64_t payload_total = 0;
        for (const auto& r : group.records) {
            uint64_t s = r.size();
            if (payload_total > std::numeric_limits<uint64_t>::max() - s) throw std::runtime_error("payload size overflow");
            payload_total += s;
        }
        uint64_t group_total = static_cast<uint64_t>(bin_rec_size) + payload_total;
        auto [data_file_name, index_copy_offset] = data_writer.reserve_group(group_total);

        IndexRecordFixed rec = build_fixed_index_record(group, item_no, kGroupSize, kFileNameBytes, data_file_name, index_copy_offset);
        std::vector<uint8_t> binary_copy = encode_binary_index_copy(rec, kGroupSize, kFileNameBytes);

        index_out << index_record_to_json(rec, group.dtype_name, group.seq_name).serialize(false) << '\n';

        std::ostream& dout = data_writer.stream();
        dout.write(reinterpret_cast<const char*>(binary_copy.data()), static_cast<std::streamsize>(binary_copy.size()));
        if (!dout) throw std::runtime_error("failed to write binary index copy");
        for (const auto& r : group.records) {
            if (!r.payload.empty()) {
                dout.write(reinterpret_cast<const char*>(r.payload.data()), static_cast<std::streamsize>(r.payload.size()));
                if (!dout) throw std::runtime_error("failed to write inline payload");
            } else {
                if (!r.source_path.has_value()) throw std::runtime_error("missing source path");
                copy_file_bytes(*r.source_path, dout);
            }
        }
        data_writer.advance(group_total);
        fixed_records.push_back(rec);
    }

    data_writer.close();
    index_out.close();
    write_meta(index_dir, args, fixed_records.size(), data_writer.created_names());
    if (args.verify) verify_binary_copies(index_dir, fixed_records);

    PointCloud dataset(static_cast<int>(fixed_records.size()), static_cast<int>(kIndexVectorDim));
    for (size_t i = 0; i < fixed_records.size(); ++i) {
        double* p = dataset.pointAt(static_cast<int>(i));
        for (size_t d = 0; d < kIndexVectorDim; ++d) p[d] = fixed_records[i].index_vector[d];
    }

    if (args.height < 2) throw std::runtime_error("--height must be >= 2");
    const std::vector<size_t> ratios = parse_ratios_csv(args.ratios_text);
    if (ratios.size() != args.height - 1) throw std::runtime_error("--ratios count must equal --height - 1");

    QueryHandler handler(args.force_cpu, dataset, "index_vectors", args.height, ratios);
    handler.saveIndex(tblaeq_dir.string());

    std::cout << "input records: " << input_record_count << "\n";
    std::cout << "index groups: " << fixed_records.size() << "\n";
    std::cout << "index dir: " << index_dir << "\n";
    std::cout << "tblaeq index dir: " << tblaeq_dir << "\n";
    std::cout << "binary index copy size: " << bin_rec_size << " bytes\n";
    std::cout << "data files: " << data_writer.created_names().size() << "\n";
    std::cout << "meta: " << (index_dir / "meta.json") << "\n";
    return 0;
}

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

Meta load_meta(const fs::path& output_dir) {
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
    if (m.group_size != kGroupSize) throw std::runtime_error("meta group_size_N does not match compile-time kGroupSize");
    if (m.file_name_bytes != kFileNameBytes) throw std::runtime_error("meta file_name_bytes does not match compile-time kFileNameBytes");
    size_t expected = binary_index_record_size(m.group_size, m.file_name_bytes);
    if (m.binary_index_record_size != expected) {
        std::ostringstream oss;
        oss << "binary_index_record_size=" << m.binary_index_record_size << ", expected=" << expected;
        throw std::runtime_error(oss.str());
    }
    return m;
}

uint8_t checked_u8(int64_t v, const std::string& key) {
    if (v < 0 || v > 255) throw std::runtime_error("field does not fit uint8: " + key);
    return static_cast<uint8_t>(v);
}

void set_file_name(IndexRecordFixed& rec, const std::string& name, size_t file_name_bytes) {
    if (file_name_bytes > MAX_FILE_NAME_BYTES) throw std::runtime_error("file_name_bytes exceeds runtime maximum");
    if (name.size() > file_name_bytes) throw std::runtime_error("file_name too long for fixed field: " + name);
    std::memset(rec.file_name, 0, sizeof(rec.file_name));
    std::memcpy(rec.file_name, name.data(), name.size());
}

IndexRecordFixed parse_index_line(const std::string& line, size_t line_no, const Meta& meta) {
    picojson::value root;
    std::string err = picojson::parse(root, line);
    if (!err.empty()) throw std::runtime_error("failed to parse index JSONL line " + std::to_string(line_no) + ": " + err);
    const auto& obj = require_object(root, "index JSONL line");

    IndexRecordFixed rec{};
    rec.item_no = json_u64(obj, "item_no");
    rec.seq = checked_u8(json_i64(obj, "seq"), "seq");
    rec.dtype = checked_u8(json_i64(obj, "dtype_id"), "dtype_id");
    rec.avg_timestamp = json_double(obj, "avg_timestamp");
    rec.avg_timestamp_us = json_i64(obj, "avg_timestamp_us");
    rec.slot_count = checked_u8(json_i64(obj, "slot_count"), "slot_count");
    rec.valid_count = checked_u8(json_i64(obj, "valid_count"), "valid_count");
    if (rec.slot_count != meta.group_size) throw std::runtime_error("slot_count does not match meta at line " + std::to_string(line_no));
    if (rec.valid_count > rec.slot_count) throw std::runtime_error("valid_count > slot_count at line " + std::to_string(line_no));
    rec.index_copy_offset = json_u64(obj, "index_copy_offset");
    rec.binary_index_record_size = json_u64(obj, "binary_index_record_size");
    if (rec.binary_index_record_size != meta.binary_index_record_size) throw std::runtime_error("binary_index_record_size mismatch at line " + std::to_string(line_no));
    set_file_name(rec, json_string(obj, "file_name"), meta.file_name_bytes);

    const auto& vec = json_array(obj, "index_vector");
    if (vec.size() != 4) throw std::runtime_error("index_vector length must be 4 at line " + std::to_string(line_no));
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

    if (rec.data_offsets[0] != 0) throw std::runtime_error("data_offsets[0] must be 0 at line " + std::to_string(line_no));
    for (size_t i = 0; i < n; ++i) {
        if (rec.data_offsets[i + 1] < rec.data_offsets[i]) throw std::runtime_error("data_offsets not monotonic at line " + std::to_string(line_no));
        const uint64_t derived_size = rec.data_offsets[i + 1] - rec.data_offsets[i];
        if (derived_size != rec.sizes[i]) throw std::runtime_error("sizes[i] != data_offsets[i+1]-data_offsets[i] at line " + std::to_string(line_no));
        if (i >= rec.valid_count && rec.sizes[i] != 0) throw std::runtime_error("padding slot has non-zero size at line " + std::to_string(line_no));
    }
    return rec;
}

std::vector<IndexRecordFixed> load_index_records(const fs::path& output_dir, const Meta& meta) {
    const fs::path index_path = output_dir / meta.index_file;
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

std::vector<SlotRef> build_slots_for_candidates(const std::vector<IndexRecordFixed>& records,
                                                const std::vector<size_t>& candidates,
                                                bool include_padding) {
    std::vector<SlotRef> slots;
    for (size_t ri : candidates) {
        if (ri >= records.size()) throw std::runtime_error("candidate index out of range: " + std::to_string(ri));
        const auto& rec = records[ri];
        for (size_t slot = 0; slot < kGroupSize; ++slot) {
            uint64_t size = rec.data_offsets[slot + 1] - rec.data_offsets[slot];
            if (!include_padding && size == 0) continue;
            slots.push_back(SlotRef{ri, static_cast<uint8_t>(slot), rec.timestamp_us[slot], rec.timestamps[slot], size});
        }
    }
    std::sort(slots.begin(), slots.end(), [](const SlotRef& a, const SlotRef& b) {
        if (a.timestamp_us != b.timestamp_us) return a.timestamp_us < b.timestamp_us;
        if (a.record_idx != b.record_idx) return a.record_idx < b.record_idx;
        return a.slot < b.slot;
    });
    return slots;
}

std::vector<Match> sort_limit_by_diff(std::vector<Match> m, size_t top_k) {
    std::sort(m.begin(), m.end(), [](const Match& a, const Match& b) {
        if (a.diff_us != b.diff_us) return a.diff_us < b.diff_us;
        if (a.slot_ref.timestamp_us != b.slot_ref.timestamp_us) return a.slot_ref.timestamp_us < b.slot_ref.timestamp_us;
        if (a.slot_ref.record_idx != b.slot_ref.record_idx) return a.slot_ref.record_idx < b.slot_ref.record_idx;
        return a.slot_ref.slot < b.slot_ref.slot;
    });
    if (top_k > 0 && m.size() > top_k) m.resize(top_k);
    return m;
}

std::vector<Match> query_interval(const std::vector<SlotRef>& slots, int64_t lo, int64_t hi, std::optional<int64_t> center, size_t top_k) {
    auto lower = std::lower_bound(slots.begin(), slots.end(), lo, [](const SlotRef& s, int64_t v) { return s.timestamp_us < v; });
    auto upper = std::upper_bound(slots.begin(), slots.end(), hi, [](int64_t v, const SlotRef& s) { return v < s.timestamp_us; });
    std::vector<Match> matches;
    for (auto it = lower; it != upper; ++it) {
        int64_t diff = center.has_value() ? std::llabs(it->timestamp_us - *center) : 0;
        matches.push_back(Match{*it, diff});
    }
    if (center.has_value()) return sort_limit_by_diff(std::move(matches), top_k);
    std::sort(matches.begin(), matches.end(), [](const Match& a, const Match& b) {
        if (a.slot_ref.timestamp_us != b.slot_ref.timestamp_us) return a.slot_ref.timestamp_us < b.slot_ref.timestamp_us;
        if (a.slot_ref.record_idx != b.slot_ref.record_idx) return a.slot_ref.record_idx < b.slot_ref.record_idx;
        return a.slot_ref.slot < b.slot_ref.slot;
    });
    if (top_k > 0 && matches.size() > top_k) matches.resize(top_k);
    return matches;
}

std::vector<Match> query_nearest(const std::vector<SlotRef>& slots, int64_t target, size_t top_k) {
    if (slots.empty()) return {};
    auto right = std::lower_bound(slots.begin(), slots.end(), target, [](const SlotRef& s, int64_t v) { return s.timestamp_us < v; });
    auto left = right;
    bool has_left = left != slots.begin();
    if (has_left) --left;
    bool has_right = right != slots.end();
    std::vector<Match> matches;
    const size_t desired = top_k == 0 ? std::numeric_limits<size_t>::max() : top_k;
    int64_t worst_kept = std::numeric_limits<int64_t>::max();
    while (has_left || has_right) {
        int64_t ld = has_left ? std::llabs(left->timestamp_us - target) : std::numeric_limits<int64_t>::max();
        int64_t rd = has_right ? std::llabs(right->timestamp_us - target) : std::numeric_limits<int64_t>::max();
        bool take_left = ld <= rd;
        SlotRef cur = take_left ? *left : *right;
        int64_t diff = take_left ? ld : rd;
        if (top_k > 0 && matches.size() >= desired && diff > worst_kept) {
            int64_t other = take_left ? rd : ld;
            if (other > worst_kept) break;
        }
        matches.push_back(Match{cur, diff});
        matches = sort_limit_by_diff(std::move(matches), top_k);
        if (top_k > 0 && matches.size() == desired) worst_kept = matches.back().diff_us;
        if (take_left) {
            if (left == slots.begin()) has_left = false;
            else --left;
        } else {
            ++right;
            has_right = right != slots.end();
        }
        if (top_k == 0 && !has_left && !has_right) break;
    }
    return sort_limit_by_diff(std::move(matches), top_k);
}

std::vector<uint8_t> read_payload_checked(const fs::path& output_dir, const IndexRecordFixed& rec, uint8_t slot) {
    if (slot >= rec.slot_count) throw std::runtime_error("slot out of range");
    uint64_t start_rel = rec.data_offsets[slot];
    uint64_t end_rel = rec.data_offsets[slot + 1];
    if (end_rel < start_rel) throw std::runtime_error("invalid data_offsets for slot");
    uint64_t size = end_rel - start_rel;
    uint64_t data_base = rec.index_copy_offset + rec.binary_index_record_size;
    if (data_base < rec.index_copy_offset) throw std::runtime_error("data_base overflow");
    uint64_t abs_start = data_base + start_rel;
    if (abs_start < data_base) throw std::runtime_error("payload offset overflow");
    const fs::path data_path = output_dir / fixed_file_name_to_string(rec.file_name);
    if (!fs::exists(data_path)) throw std::runtime_error("data file not found: " + data_path.string());
    uint64_t file_size = static_cast<uint64_t>(fs::file_size(data_path));
    if (abs_start > file_size || size > file_size - abs_start) {
        std::ostringstream oss;
        oss << "payload range exceeds data file length: file=" << data_path
            << " abs_start=" << abs_start << " size=" << size << " file_size=" << file_size;
        throw std::runtime_error(oss.str());
    }
    std::vector<uint8_t> payload(size);
    if (size == 0) return payload;
    std::ifstream in(data_path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open data file: " + data_path.string());
    in.seekg(static_cast<std::streamoff>(abs_start), std::ios::beg);
    in.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    if (in.gcount() != static_cast<std::streamsize>(payload.size())) throw std::runtime_error("short payload read after length check");
    return payload;
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

void dump_payloads(const fs::path& index_dir, const fs::path& dump_dir,
                   const std::vector<IndexRecordFixed>& records,
                   const std::vector<Match>& matches) {
    fs::create_directories(dump_dir);
    for (size_t rank = 0; rank < matches.size(); ++rank) {
        const auto& s = matches[rank].slot_ref;
        const auto& rec = records[s.record_idx];
        std::vector<uint8_t> payload = read_payload_checked(index_dir, rec, s.slot);
        std::ostringstream name;
        name << "match_" << std::setw(4) << std::setfill('0') << rank
             << "_item" << std::setw(8) << std::setfill('0') << rec.item_no
             << "_slot" << std::setw(3) << std::setfill('0') << static_cast<int>(s.slot)
             << "_seq" << std::setw(4) << std::setfill('0') << static_cast<int>(rec.seq)
             << "_dtype" << static_cast<int>(rec.dtype)
             << "_" << safe_timestamp(s.timestamp) << extension_for_type(rec.dtype);
        fs::path out_path = dump_dir / name.str();
        std::ofstream out(out_path, std::ios::binary);
        if (!out) throw std::runtime_error("failed to open dump file: " + out_path.string());
        if (!payload.empty()) out.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
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

std::vector<size_t> extract_candidate_indices(const QueryResult& result) {
    if (result.errorCode != 0) throw std::runtime_error("T-BLAEQ query failed with error code " + std::to_string(result.errorCode));
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

int run_query(QueryArgs args) {
    fs::path index_root(args.index_root);
    fs::path index_dir = index_root / args.index_dir;
    fs::path tblaeq_dir = index_root / args.tblaeq_dir;
    if (!fs::exists(index_dir)) throw std::runtime_error("index dir not found: " + index_dir.string());
    if (!fs::exists(tblaeq_dir)) throw std::runtime_error("tblaeq index dir not found: " + tblaeq_dir.string());

    Meta meta = load_meta(index_dir);
    std::vector<IndexRecordFixed> records = load_index_records(index_dir, meta);

    QueryHandler handler(tblaeq_dir.string(), /*loadFromIndex=*/true);
    handler.prepareForQuery(LevelPolicy::L3);

    std::vector<size_t> candidates;
    if (args.mode == "knn") {
        if (!args.has_timestamp) throw std::runtime_error("--timestamp is required for knn mode");
        if (!args.has_seq) throw std::runtime_error("--seq is required for knn mode");
        if (args.knn_k == 0) throw std::runtime_error("--knn-k must be > 0 for knn mode");
        auto v3 = split_timestamp_3d(args.timestamp);
        std::vector<double> query_point = {v3[0], v3[1], v3[2], static_cast<double>(args.seq)};
        QueryResult result = handler.performSingleKNNQuery(query_point, args.knn_k, /*saveFineMesh=*/true);
        candidates = extract_candidate_indices(result);
    } else if (args.mode == "range") {
        if (!args.has_start || !args.has_end) throw std::runtime_error("--start and --end are required for range mode");
        if (!args.has_seq) throw std::runtime_error("--seq is required for range mode");
        if (args.end < args.start) throw std::runtime_error("--end must be >= --start");
        auto v3_start = split_timestamp_3d(args.start);
        auto v3_end = split_timestamp_3d(args.end);
        std::vector<double> lower(4);
        std::vector<double> upper(4);
        for (size_t i = 0; i < 3; ++i) {
            lower[i] = std::min(v3_start[i], v3_end[i]);
            upper[i] = std::max(v3_start[i], v3_end[i]);
        }
        lower[3] = static_cast<double>(args.seq);
        upper[3] = static_cast<double>(args.seq);
        QueryResult result = handler.performSingleRangeQuery(upper, lower, /*saveFineMesh=*/true);
        candidates = extract_candidate_indices(result);
    } else {
        throw std::runtime_error("unsupported mode: " + args.mode);
    }

    std::vector<SlotRef> slots = build_slots_for_candidates(records, candidates, args.include_padding);
    std::vector<Match> matches;
    if (args.mode == "knn") {
        int64_t target = timestamp_to_us(args.timestamp);
        matches = query_nearest(slots, target, args.knn_k);
    } else {
        int64_t lo = timestamp_to_us(args.start);
        int64_t hi = timestamp_to_us(args.end);
        matches = query_interval(slots, lo, hi, std::nullopt, args.range_limit);
    }

    std::cout << "Loaded index records: " << records.size() << "\n";
    std::cout << "Coarse candidates: " << candidates.size() << "\n";
    std::cout << "Fine slots: " << slots.size() << "\n";
    std::cout << "Matches: " << matches.size() << "\n";
    print_matches(records, matches);

    fs::path dump_dir = args.has_dump_dir ? fs::path(args.dump_dir) : (index_root / "payloads");
    dump_payloads(index_dir, dump_dir, records, matches);
    std::cout << "Dumped " << matches.size() << " payload(s) to: " << dump_dir << "\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    BuildArgs build_args;
    QueryArgs query_args;

    CLI::App app{"Build JSONL index + T-BLAEQ index, then query both."};
    auto* build = app.add_subcommand("build", "Build JSONL index + T-BLAEQ index");
    auto* query = app.add_subcommand("query", "Query both indexes (T-BLAEQ coarse + JSONL refine)");
    app.require_subcommand(1);

    build->add_option("root", build_args.root, "input dataset root directory")->required()->check(CLI::ExistingDirectory);
    build->add_option("csv", build_args.csv_path, "coordinate CSV path")->required()->check(CLI::ExistingFile);
    build->add_option("index_root", build_args.index_root, "index root directory")->required();
    build->add_option("--index-dir", build_args.index_dir, "index file directory name")->default_val(kDefaultIndexDir);
    build->add_option("--tblaeq-dir", build_args.tblaeq_dir, "T-BLAEQ index directory name")->default_val(kDefaultTblaeqDir);
    build->add_option("--index-file", build_args.index_file, "main JSONL index file name")->default_val("index.jsonl");
    build->add_option("--max-data-file-size", build_args.max_data_file_size_text, "max data file size, e.g. 1GiB")->default_val("1GiB");
    build->add_flag("--verify", build_args.verify, "verify binary index copies after build");
    build->add_flag("--force-cpu", build_args.force_cpu, "force CPU index building");
    build->add_option("--height", build_args.height, "hierarchy height (levels), must be >= 2")->default_val(IndexBuilder::kDefaultHeight);
    build->add_option("--ratios", build_args.ratios_text, "comma-separated coarsening ratios (height-1 entries)")
        ->default_val("100,50,20");

    std::string seq_filter;
    query->add_option("index_root", query_args.index_root, "index root directory")->required()->check(CLI::ExistingDirectory);
    query->add_option("--index-dir", query_args.index_dir, "index file directory name")->default_val(kDefaultIndexDir);
    query->add_option("--tblaeq-dir", query_args.tblaeq_dir, "T-BLAEQ index directory name")->default_val(kDefaultTblaeqDir);
    query->add_option("--mode", query_args.mode, "knn or range")->default_val("knn")->check(CLI::IsMember({"knn", "range"}));
    auto* opt_timestamp = query->add_option("--timestamp", query_args.timestamp, "query timestamp (knn mode)");
    auto* opt_start = query->add_option("--start", query_args.start, "range start timestamp");
    auto* opt_end = query->add_option("--end", query_args.end, "range end timestamp");
    auto* opt_seq = query->add_option("--seq", seq_filter, "sequence filter, accepts 1 or seq0001");
    query->add_option("--knn-k", query_args.knn_k, "K for knn mode")->default_val(10);
    query->add_option("--range-limit", query_args.range_limit, "max returned matches in range mode (0 means all)")->default_val(0);
    query->add_flag("--include-padding", query_args.include_padding, "include zero-size padded slots");
    auto* opt_dump = query->add_option("--dump-dir", query_args.dump_dir, "optional directory to dump matched payload bytes");

    CLI11_PARSE(app, argc, argv);

    try {
        if (*build) {
            build_args.max_data_file_size = parse_size_arg(build_args.max_data_file_size_text);
            return run_build(build_args);
        }
        if (*query) {
            query_args.has_timestamp = opt_timestamp->count() > 0;
            query_args.has_start = opt_start->count() > 0;
            query_args.has_end = opt_end->count() > 0;
            query_args.has_seq = opt_seq->count() > 0;
            query_args.has_dump_dir = opt_dump->count() > 0;
            if (query_args.has_seq) query_args.seq = parse_seq_arg(seq_filter);
            return run_query(query_args);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
