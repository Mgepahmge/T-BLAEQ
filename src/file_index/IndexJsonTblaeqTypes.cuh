#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace tblaeq {
namespace file_index {

namespace fs = std::filesystem;

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
    size_t height = 0;
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
    std::vector<uint8_t> seqs;
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

} // namespace file_index
} // namespace tblaeq
