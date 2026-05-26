#include "IndexJsonTblaeqBuild.cuh"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "IndexJsonTblaeqPicojson.cuh"
#include "IndexJsonTblaeqUtils.cuh"
#include "src/core/QueryHandler.cuh"

namespace tblaeq {
namespace file_index {

namespace {
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
                    if (i + 1 < line.size() && line[i + 1] == '"') {
                        cur.push_back('"');
                        ++i;
                    }
                    else in_quotes = false;
                }
                else cur.push_back(c);
            }
            else {
                if (c == '"') in_quotes = true;
                else if (c == ',') {
                    fields.push_back(cur);
                    cur.clear();
                }
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
                if (i >= fields.size()) throw std::runtime_error(
                    "CSV row missing field at row " + std::to_string(row_no));
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
            if (cls < 0 || cls > 255) throw std::runtime_error(
                "Classification does not fit uint8 at row " + std::to_string(row_no));
            rec.payload = encode_coord_payload(pos[0], pos[1], pos[2], static_cast<uint8_t>(cls));
            records.push_back(std::move(rec));
        }
    }

    std::vector<IndexGroup> make_groups(std::vector<SourceRecord> records, size_t group_size) {
        std::map<std::pair<int, int>, std::vector<SourceRecord>> buckets;
        for (auto& rec : records) buckets[{static_cast<int>(rec.seq), static_cast<int>(rec.dtype)}].push_back(
            std::move(rec));
        std::vector<IndexGroup> groups;
        for (auto& kv : buckets) {
            auto& bucket = kv.second;
            std::sort(bucket.begin(), bucket.end(), [](const auto& a, const auto& b) {
                return a.timestamp < b.timestamp;
            });
            for (size_t start = 0; start < bucket.size(); start += group_size) {
                size_t end = std::min(bucket.size(), start + group_size);
                IndexGroup g;
                g.records.reserve(end - start);
                double sum = 0.0;
                for (size_t i = start; i < end; ++i) {
                    sum += bucket[i].timestamp;
                    g.records.push_back(std::move(bucket[i]));
                }
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
        if (v > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) throw std::runtime_error(
            "uint64 too large for JSON int64");
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

    picojson::value index_record_to_json(const IndexRecordFixed& rec, const std::string& dtype_name,
                                         const std::string& seq_name) {
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

        for (size_t i = 0; i < group.records.size(); ++i) {
            rec.timestamps[i] = group.records[i].timestamp;
            rec.timestamp_us[i] = timestamp_to_us(group.records[i].timestamp);
            rec.sizes[i] = group.records[i].size();
        }
        const double last_ts = rec.timestamps[group.records.size() - 1];
        const int64_t last_ts_us = rec.timestamp_us[group.records.size() - 1];
        for (size_t i = group.records.size(); i < group_size; ++i) {
            rec.timestamps[i] = last_ts;
            rec.timestamp_us[i] = last_ts_us;
            rec.sizes[i] = 0;
        }
        uint64_t total = 0;
        rec.data_offsets[0] = 0;
        for (size_t i = 0; i < group_size; ++i) {
            if (total > std::numeric_limits<uint64_t>::max() - rec.sizes[i]) throw std::runtime_error(
                "data offset overflow");
            total += rec.sizes[i];
            rec.data_offsets[i + 1] = total;
        }
        return rec;
    }

    std::vector<uint8_t> encode_binary_index_copy(const IndexRecordFixed& rec, size_t n, size_t file_name_bytes) {
        if (n == 0 || n > MAX_N) throw std::runtime_error("invalid N while encoding binary index");
        if (file_name_bytes == 0 || file_name_bytes > MAX_FILE_NAME_BYTES) throw std::runtime_error(
            "invalid file_name_bytes");
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
        if (out.size() != binary_index_record_size(n, file_name_bytes)) throw std::runtime_error(
            "binary index size mismatch");
        return out;
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
            : output_dir_(std::move(output_dir)), max_size_(max_size), file_name_bytes_(file_name_bytes) {
        }

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
            if (in.gcount() != static_cast<std::streamsize>(actual.size())) throw std::runtime_error(
                "failed to read binary index copy");
            if (actual != expected) throw std::runtime_error(
                "binary index copy mismatch at item " + std::to_string(rec.item_no));
        }
    }

    void write_meta(const fs::path& output_dir, const BuildArgs& args, size_t record_count,
                    const std::vector<std::string>& data_files) {
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
        meta["data_access_rule"] = picojson::value(
            "data_base = index_copy_offset + binary_index_record_size; payload_start = data_base + data_offsets[i]; payload_end = data_base + data_offsets[i+1]");
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
} // namespace

int run_build(BuildArgs args) {
    fs::path root(args.root);
    fs::path index_root(args.index_root);
    if (!fs::exists(root) || !fs::is_directory(root)) throw std::runtime_error(
        "root is not a directory: " + root.string());
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
    if (!index_out) throw std::runtime_error(
        "failed to write main index: " + (index_dir / args.index_file).string());

    DataFileWriter data_writer(index_dir, args.max_data_file_size, kFileNameBytes);
    std::vector<IndexRecordFixed> verify_records;
    if (args.verify) verify_records.reserve(groups.size());

    PointCloud dataset(static_cast<int>(groups.size()), static_cast<int>(kIndexVectorDim));

    for (size_t item_no = 0; item_no < groups.size(); ++item_no) {
        auto& group = groups[item_no];
        std::sort(group.records.begin(), group.records.end(), [](const auto& a, const auto& b) {
            return a.timestamp < b.timestamp;
        });
        uint64_t payload_total = 0;
        for (const auto& r : group.records) {
            uint64_t s = r.size();
            if (payload_total > std::numeric_limits<uint64_t>::max() - s) throw std::runtime_error(
                "payload size overflow");
            payload_total += s;
        }
        uint64_t group_total = static_cast<uint64_t>(bin_rec_size) + payload_total;
        auto [data_file_name, index_copy_offset] = data_writer.reserve_group(group_total);

        IndexRecordFixed rec = build_fixed_index_record(group, item_no, kGroupSize, kFileNameBytes, data_file_name,
                                                        index_copy_offset);
        std::vector<uint8_t> binary_copy = encode_binary_index_copy(rec, kGroupSize, kFileNameBytes);

        index_out << index_record_to_json(rec, group.dtype_name, group.seq_name).serialize(false) << '\n';

        std::ostream& dout = data_writer.stream();
        dout.write(reinterpret_cast<const char*>(binary_copy.data()),
                   static_cast<std::streamsize>(binary_copy.size()));
        if (!dout) throw std::runtime_error("failed to write binary index copy");
        for (const auto& r : group.records) {
            if (!r.payload.empty()) {
                dout.write(reinterpret_cast<const char*>(r.payload.data()),
                           static_cast<std::streamsize>(r.payload.size()));
                if (!dout) throw std::runtime_error("failed to write inline payload");
            }
            else {
                if (!r.source_path.has_value()) throw std::runtime_error("missing source path");
                copy_file_bytes(*r.source_path, dout);
            }
        }
        data_writer.advance(group_total);
        if (args.verify) verify_records.push_back(rec);

        double* p = dataset.pointAt(static_cast<int>(item_no));
        for (size_t d = 0; d < kIndexVectorDim; ++d) p[d] = rec.index_vector[d];
    }

    data_writer.close();
    index_out.close();
    write_meta(index_dir, args, groups.size(), data_writer.created_names());
    if (args.verify) verify_binary_copies(index_dir, verify_records);

    if (args.height < 2) throw std::runtime_error("--height must be >= 2");
    const std::vector<size_t> ratios = parse_ratios_csv(args.ratios_text);
    if (ratios.size() != args.height - 1) throw std::runtime_error("--ratios count must equal --height - 1");

    QueryHandler handler(args.force_cpu, dataset, "index_vectors", args.height, ratios);
    handler.saveIndex(tblaeq_dir.string());

    std::cout << "input records: " << input_record_count << "\n";
    std::cout << "index groups: " << groups.size() << "\n";
    std::cout << "index dir: " << index_dir << "\n";
    std::cout << "tblaeq index dir: " << tblaeq_dir << "\n";
    std::cout << "binary index copy size: " << bin_rec_size << " bytes\n";
    std::cout << "data files: " << data_writer.created_names().size() << "\n";
    std::cout << "meta: " << (index_dir / "meta.json") << "\n";
    return 0;
}

} // namespace file_index
} // namespace tblaeq
