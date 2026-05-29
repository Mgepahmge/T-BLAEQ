#include "src/file_index/IndexJsonTblaeqBuild.cuh"
#include "src/file_index/IndexJsonTblaeqJson.cuh"
#include "src/file_index/IndexJsonTblaeqPicojson.cuh"
#include "src/file_index/IndexJsonTblaeqQuery.cuh"
#include "src/file_index/IndexJsonTblaeqUtils.cuh"
#include "src/core/MemoryPolicy.cuh"
#include "src/core/QueryHandler.cuh"

#include "CLI11.hpp"
#include "httplib.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr const char* kDefaultHost = "0.0.0.0";
constexpr int kDefaultPort = 8090;
constexpr const char* kDefaultIndexCacheDir = "./tempJsonIndexCache";

std::string dataset_root_path(const std::string& dataset_name, const std::string& base_path) {
    return (fs::path(base_path) / dataset_name).string();
}

bool is_valid_dataset_name(const std::string& dataset_name) {
    static const std::regex kDatasetNamePattern("^[A-Za-z0-9._-]+$");
    return !dataset_name.empty() && std::regex_match(dataset_name, kDatasetNamePattern);
}

bool is_valid_data_file(const std::string& file_name) {
    static const std::regex kFileNamePattern("^[A-Za-z0-9._-]+$");
    return !file_name.empty() && std::regex_match(file_name, kFileNamePattern);
}

bool index_exists(const std::string& dataset_name, const std::string& base_path) {
    const fs::path root = dataset_root_path(dataset_name, base_path);
    const fs::path index_dir = root / tblaeq::file_index::kDefaultIndexDir;
    const fs::path tblaeq_dir = root / tblaeq::file_index::kDefaultTblaeqDir;
    return fs::exists(index_dir / "meta.json") && fs::exists(tblaeq_dir / "metadata.bin");
}

std::vector<std::string> list_indexed_datasets(const std::string& base_path) {
    std::vector<std::string> datasets;
    const fs::path base(base_path);
    if (!fs::exists(base) || !fs::is_directory(base)) {
        return datasets;
    }

    for (const auto& entry : fs::directory_iterator(base)) {
        if (!entry.is_directory()) continue;
        const auto name = entry.path().filename().string();
        if (index_exists(name, base_path)) datasets.push_back(name);
    }
    std::sort(datasets.begin(), datasets.end());
    return datasets;
}

void set_json_response(httplib::Response& res, const picojson::object& body, int status_code = 200) {
    res.status = status_code;
    res.set_header("Content-Type", "application/json");
    res.set_content(picojson::value(body).serialize(), "application/json");
}

void send_error(httplib::Response& res, int status_code, const std::string& message) {
    picojson::object body;
    body["success"] = picojson::value(false);
    body["error"] = picojson::value(message);
    set_json_response(res, body, status_code);
}

bool parse_json_body(const httplib::Request& req, picojson::object& out, std::string& err) {
    if (req.body.empty()) {
        err = "request body is empty";
        return false;
    }
    picojson::value root;
    std::string parse_err = picojson::parse(root, req.body);
    if (!parse_err.empty()) {
        err = "invalid JSON: " + parse_err;
        return false;
    }
    if (!root.is<picojson::object>()) {
        err = "JSON body must be an object";
        return false;
    }
    out = root.get<picojson::object>();
    return true;
}

bool parse_string_field(const picojson::object& obj, const std::string& key, std::string& out,
                        bool required, std::string& err) {
    auto it = obj.find(key);
    if (it == obj.end()) {
        if (required) err = "missing field: " + key;
        return !required;
    }
    if (!it->second.is<std::string>()) {
        err = "field '" + key + "' must be a string";
        return false;
    }
    out = it->second.get<std::string>();
    return true;
}

bool parse_bool_field(const picojson::object& obj, const std::string& key, bool& out,
                      bool required, std::string& err) {
    auto it = obj.find(key);
    if (it == obj.end()) {
        if (required) err = "missing field: " + key;
        return !required;
    }
    if (!it->second.is<bool>()) {
        err = "field '" + key + "' must be a boolean";
        return false;
    }
    out = it->second.get<bool>();
    return true;
}

bool parse_number_field(const picojson::object& obj, const std::string& key, double& out,
                        bool required, std::string& err) {
    auto it = obj.find(key);
    if (it == obj.end()) {
        if (required) err = "missing field: " + key;
        return !required;
    }
    if (!it->second.is<double>()) {
        err = "field '" + key + "' must be a number";
        return false;
    }
    out = it->second.get<double>();
    return true;
}

bool parse_size_field(const picojson::object& obj, const std::string& key, size_t& out,
                      bool required, std::string& err) {
    double value = 0.0;
    if (!parse_number_field(obj, key, value, required, err)) return false;
    if (!required && obj.find(key) == obj.end()) return true;
    if (value < 0 || value > static_cast<double>(std::numeric_limits<size_t>::max())) {
        err = "field '" + key + "' is out of range";
        return false;
    }
    out = static_cast<size_t>(value);
    return true;
}

bool parse_uint64(const std::string& text, uint64_t& out) {
    try {
        size_t idx = 0;
        unsigned long long value = std::stoull(text, &idx);
        if (idx != text.size()) return false;
        out = static_cast<uint64_t>(value);
        return true;
    }
    catch (const std::exception&) {
        return false;
    }
}

bool parse_seqs_field(const picojson::object& obj, std::vector<uint8_t>& out, std::string& err) {
    auto it = obj.find("seq");
    if (it == obj.end()) {
        err = "missing field: seq";
        return false;
    }
    try {
        if (it->second.is<std::string>()) {
            std::vector<std::string> items = {it->second.get<std::string>()};
            out = tblaeq::file_index::parse_seq_list(items);
            return true;
        }
        if (it->second.is<double>()) {
            double value = it->second.get<double>();
            if (value < 0 || value > 255.0) {
                err = "seq value out of range";
                return false;
            }
            out = {static_cast<uint8_t>(value)};
            return true;
        }
        if (it->second.is<picojson::array>()) {
            const auto& arr = it->second.get<picojson::array>();
            for (const auto& entry : arr) {
                if (entry.is<double>()) {
                    double value = entry.get<double>();
                    if (value < 0 || value > 255.0) {
                        err = "seq value out of range";
                        return false;
                    }
                    out.push_back(static_cast<uint8_t>(value));
                }
                else if (entry.is<std::string>()) {
                    std::vector<std::string> items = {entry.get<std::string>()};
                    std::vector<uint8_t> parsed = tblaeq::file_index::parse_seq_list(items);
                    out.insert(out.end(), parsed.begin(), parsed.end());
                }
                else {
                    err = "seq array entries must be numbers or strings";
                    return false;
                }
            }
            return true;
        }
        err = "field 'seq' must be a string, number, or array";
        return false;
    }
    catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

picojson::value to_json_int64(uint64_t value) {
    if (value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return picojson::value(static_cast<double>(value));
    }
    return picojson::value(static_cast<int64_t>(value));
}

class JsonIndexManager {
public:
    struct HandlerEntry {
        std::unique_ptr<QueryHandler> handler;
        std::mutex mutex;
    };

    explicit JsonIndexManager(std::string base_path) : base_path_(std::move(base_path)) {}

    bool exists_on_disk(const std::string& dataset_name) const {
        return index_exists(dataset_name, base_path_);
    }

    std::vector<std::string> list_datasets() const {
        return list_indexed_datasets(base_path_);
    }

    std::string dataset_root(const std::string& dataset_name) const {
        return dataset_root_path(dataset_name, base_path_);
    }

    fs::path index_dir(const std::string& dataset_name) const {
        return fs::path(base_path_) / dataset_name / tblaeq::file_index::kDefaultIndexDir;
    }

    fs::path tblaeq_dir(const std::string& dataset_name) const {
        return fs::path(base_path_) / dataset_name / tblaeq::file_index::kDefaultTblaeqDir;
    }

    fs::path data_file_path(const std::string& dataset_name, const std::string& data_file) const {
        return index_dir(dataset_name) / data_file;
    }

    void build_and_save(const std::string& dataset_name,
                        const std::string& dataset_root,
                        const std::string& csv_path,
                        const tblaeq::file_index::BuildArgs& options) {
        tblaeq::file_index::BuildArgs args = options;
        args.root = dataset_root;
        args.csv_path = csv_path;
        args.index_root = dataset_root_path(dataset_name, base_path_);
        tblaeq::file_index::run_build(args);
        drop_handler(dataset_name);
    }

    std::shared_ptr<HandlerEntry> get_or_load(const std::string& dataset_name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = handlers_.find(dataset_name);
        if (it != handlers_.end()) return it->second;

        if (!exists_on_disk(dataset_name)) {
            throw std::runtime_error("dataset index does not exist: " + dataset_name);
        }

        auto handler = std::make_unique<QueryHandler>(tblaeq_dir(dataset_name).string(), true);
        handler->prepareForQuery(LevelPolicy::L3);
        auto entry = std::make_shared<HandlerEntry>();
        entry->handler = std::move(handler);
        handlers_[dataset_name] = entry;
        return entry;
    }

    void drop_handler(const std::string& dataset_name) {
        std::lock_guard<std::mutex> lock(mutex_);
        handlers_.erase(dataset_name);
    }

private:
    std::string base_path_;
    std::unordered_map<std::string, std::shared_ptr<HandlerEntry>> handlers_;
    std::mutex mutex_;
};

bool stream_payload(JsonIndexManager& manager,
                    const std::string& dataset_name,
                    const std::string& data_file,
                    uint64_t offset,
                    uint64_t size,
                    httplib::Response& res,
                    int& status,
                    std::string& err) {
    if (!is_valid_dataset_name(dataset_name)) {
        err = "invalid dataset_name";
        status = 400;
        return false;
    }
    if (!is_valid_data_file(data_file)) {
        err = "invalid data_file";
        status = 400;
        return false;
    }

    const fs::path data_path = manager.data_file_path(dataset_name, data_file);
    if (!fs::exists(data_path)) {
        err = "payload file not found";
        status = 404;
        return false;
    }

    uint64_t file_size = static_cast<uint64_t>(fs::file_size(data_path));
    if (offset > file_size || size > file_size - offset) {
        err = "payload range out of bounds";
        status = 416;
        return false;
    }

    if (size == 0) {
        res.status = 200;
        res.set_header("Content-Type", "application/octet-stream");
        res.set_content("", "application/octet-stream");
        return true;
    }

    auto file = std::make_shared<std::ifstream>(data_path, std::ios::binary);
    if (!*file) {
        err = "failed to open payload file";
        status = 500;
        return false;
    }

    res.status = 200;
    res.set_header("Content-Type", "application/octet-stream");
    res.set_header("Content-Length", std::to_string(size));
    res.set_content_provider(
        size,
        "application/octet-stream",
        [file, offset](size_t out_offset, size_t length, httplib::DataSink& sink) {
            if (!file->good()) return false;
            const uint64_t abs = offset + static_cast<uint64_t>(out_offset);
            file->seekg(static_cast<std::streamoff>(abs), std::ios::beg);
            if (!file->good()) return false;
            std::string buffer(length, '\0');
            file->read(buffer.data(), static_cast<std::streamsize>(length));
            std::streamsize got = file->gcount();
            if (got <= 0) return false;
            sink.write(buffer.data(), static_cast<size_t>(got));
            return static_cast<size_t>(got) == length;
        },
        [file](bool) {}
    );
    return true;
}

void setup_routes(httplib::Server& server, JsonIndexManager& manager) {
    server.Post("/build-index", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            picojson::object body;
            std::string err;
            if (!parse_json_body(req, body, err)) {
                send_error(res, 400, err);
                return;
            }

            std::string dataset_name;
            if (!parse_string_field(body, "dataset_name", dataset_name, true, err)) {
                send_error(res, 400, err);
                return;
            }
            if (!is_valid_dataset_name(dataset_name)) {
                send_error(res, 400, "invalid dataset_name");
                return;
            }

            bool already_exists = manager.exists_on_disk(dataset_name);
            if (already_exists) {
                picojson::object out;
                out["success"] = picojson::value(true);
                out["dataset_name"] = picojson::value(dataset_name);
                out["already_exists"] = picojson::value(true);
                out["message"] = picojson::value("index already exists");
                set_json_response(res, out);
                return;
            }

            std::string dataset_root;
            std::string csv_path;
            if (!parse_string_field(body, "dataset_root", dataset_root, true, err) ||
                !parse_string_field(body, "csv_path", csv_path, true, err)) {
                send_error(res, 400, err);
                return;
            }

            tblaeq::file_index::BuildArgs build_args;
            if (!parse_bool_field(body, "force_cpu", build_args.force_cpu, false, err) ||
                !parse_bool_field(body, "verify", build_args.verify, false, err)) {
                send_error(res, 400, err);
                return;
            }

            size_t height = build_args.height;
            if (!parse_size_field(body, "height", height, false, err)) {
                send_error(res, 400, err);
                return;
            }
            build_args.height = height;

            std::string ratios;
            if (!parse_string_field(body, "ratios", ratios, false, err)) {
                send_error(res, 400, err);
                return;
            }
            if (!ratios.empty()) {
                build_args.ratios_text = ratios;
            }

            std::string size_text;
            if (!parse_string_field(body, "max_data_file_size", size_text, false, err)) {
                send_error(res, 400, err);
                return;
            }
            if (!size_text.empty()) {
                build_args.max_data_file_size_text = size_text;
            }
            try {
                build_args.max_data_file_size = tblaeq::file_index::parse_size_arg(build_args.max_data_file_size_text);
            }
            catch (const std::exception& e) {
                send_error(res, 400, e.what());
                return;
            }

            manager.build_and_save(dataset_name, dataset_root, csv_path, build_args);

            picojson::object out;
            out["success"] = picojson::value(true);
            out["dataset_name"] = picojson::value(dataset_name);
            out["already_exists"] = picojson::value(false);
            out["message"] = picojson::value("index build and save succeeded");
            set_json_response(res, out);
        }
        catch (const std::exception& e) {
            send_error(res, 500, e.what());
        }
    });

    auto datasets_handler = [&](const httplib::Request&, httplib::Response& res) {
        picojson::object out;
        auto datasets = manager.list_datasets();
        picojson::array arr;
        for (const auto& name : datasets) arr.push_back(picojson::value(name));
        out["success"] = picojson::value(true);
        out["datasets_count"] = to_json_int64(datasets.size());
        out["datasets"] = picojson::value(arr);
        set_json_response(res, out);
    };

    server.Get("/datasets", datasets_handler);
    server.Post("/datasets", datasets_handler);

    server.Post("/query", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            picojson::object body;
            std::string err;
            if (!parse_json_body(req, body, err)) {
                send_error(res, 400, err);
                return;
            }

            std::string dataset_name;
            if (!parse_string_field(body, "dataset_name", dataset_name, true, err)) {
                send_error(res, 400, err);
                return;
            }
            if (!is_valid_dataset_name(dataset_name)) {
                send_error(res, 400, "invalid dataset_name");
                return;
            }

            std::string mode = "knn";
            if (!parse_string_field(body, "mode", mode, false, err)) {
                send_error(res, 400, err);
                return;
            }
            if (!mode.empty()) {
                mode = tblaeq::file_index::to_lower_ascii(mode);
            }
            if (mode != "knn" && mode != "range") {
                send_error(res, 400, "unsupported mode: " + mode);
                return;
            }

            tblaeq::file_index::QueryArgs args;
            args.index_root = manager.dataset_root(dataset_name);
            args.mode = mode;

            if (!parse_seqs_field(body, args.seqs, err)) {
                send_error(res, 400, err);
                return;
            }
            args.has_seq = !args.seqs.empty();

            if (!parse_bool_field(body, "include_padding", args.include_padding, false, err)) {
                send_error(res, 400, err);
                return;
            }

            size_t knn_k = args.knn_k;
            if (!parse_size_field(body, "knn_k", knn_k, false, err)) {
                send_error(res, 400, err);
                return;
            }
            args.knn_k = knn_k;

            size_t range_limit = args.range_limit;
            if (!parse_size_field(body, "range_limit", range_limit, false, err)) {
                send_error(res, 400, err);
                return;
            }
            args.range_limit = range_limit;

            double timestamp = 0.0;
            if (body.find("timestamp") != body.end()) {
                if (!parse_number_field(body, "timestamp", timestamp, true, err)) {
                    send_error(res, 400, err);
                    return;
                }
                args.timestamp = timestamp;
                args.has_timestamp = true;
            }
            double start = 0.0;
            if (body.find("start") != body.end()) {
                if (!parse_number_field(body, "start", start, true, err)) {
                    send_error(res, 400, err);
                    return;
                }
                args.start = start;
                args.has_start = true;
            }
            double end = 0.0;
            if (body.find("end") != body.end()) {
                if (!parse_number_field(body, "end", end, true, err)) {
                    send_error(res, 400, err);
                    return;
                }
                args.end = end;
                args.has_end = true;
            }

            auto entry = manager.get_or_load(dataset_name);
            tblaeq::file_index::QueryServiceResult result;
            {
                std::unique_lock<std::mutex> lock(entry->mutex);
                result = tblaeq::file_index::run_query_service(args, *entry->handler);
            }

            bool include_payload_url = true;
            if (!parse_bool_field(body, "include_payload_url", include_payload_url, false, err)) {
                send_error(res, 400, err);
                return;
            }

            picojson::array matches;
            matches.reserve(result.matches.size());
            for (const auto& match : result.matches) {
                picojson::object m;
                m["rank"] = to_json_int64(match.rank);
                m["item_no"] = to_json_int64(match.item_no);
                m["seq"] = to_json_int64(match.seq);
                m["dtype"] = to_json_int64(match.dtype);
                m["slot"] = to_json_int64(match.slot);
                m["timestamp"] = picojson::value(match.timestamp);
                m["diff_us"] = picojson::value(static_cast<int64_t>(match.diff_us));
                m["data_file"] = picojson::value(match.data_file);
                m["payload_offset"] = to_json_int64(match.payload_offset);
                m["payload_size"] = to_json_int64(match.payload_size);
                if (include_payload_url) {
                    std::string url = "/payload?dataset_name=" + dataset_name +
                                      "&data_file=" + match.data_file +
                                      "&offset=" + std::to_string(match.payload_offset) +
                                      "&size=" + std::to_string(match.payload_size);
                    m["payload_url"] = picojson::value(url);
                }
                matches.push_back(picojson::value(m));
            }

            picojson::object out;
            out["success"] = picojson::value(true);
            out["dataset_name"] = picojson::value(dataset_name);
            out["mode"] = picojson::value(mode);
            out["candidate_count"] = to_json_int64(result.candidate_count);
            out["record_count"] = to_json_int64(result.record_count);
            out["slot_count"] = to_json_int64(result.slot_count);
            out["match_count"] = to_json_int64(result.matches.size());
            out["matches"] = picojson::value(matches);
            set_json_response(res, out);
        }
        catch (const std::exception& e) {
            send_error(res, 500, e.what());
        }
    });

    server.Get("/payload", [&](const httplib::Request& req, httplib::Response& res) {
        std::string err;
        if (!req.has_param("dataset_name") || !req.has_param("data_file") ||
            !req.has_param("offset") || !req.has_param("size")) {
            send_error(res, 400, "missing payload parameters");
            return;
        }
        const std::string dataset_name = req.get_param_value("dataset_name");
        const std::string data_file = req.get_param_value("data_file");
        uint64_t offset = 0;
        uint64_t size = 0;
        if (!parse_uint64(req.get_param_value("offset"), offset) ||
            !parse_uint64(req.get_param_value("size"), size)) {
            send_error(res, 400, "invalid offset/size");
            return;
        }
        int status = 500;
        if (!stream_payload(manager, dataset_name, data_file, offset, size, res, status, err)) {
            send_error(res, status, err);
            return;
        }
    });

    server.Post("/payload", [&](const httplib::Request& req, httplib::Response& res) {
        picojson::object body;
        std::string err;
        if (!parse_json_body(req, body, err)) {
            send_error(res, 400, err);
            return;
        }
        std::string dataset_name;
        std::string data_file;
        if (!parse_string_field(body, "dataset_name", dataset_name, true, err) ||
            !parse_string_field(body, "data_file", data_file, true, err)) {
            send_error(res, 400, err);
            return;
        }
        size_t offset = 0;
        size_t size = 0;
        if (!parse_size_field(body, "offset", offset, true, err) ||
            !parse_size_field(body, "size", size, true, err)) {
            send_error(res, 400, err);
            return;
        }
        int status = 500;
        if (!stream_payload(manager, dataset_name, data_file,
                            static_cast<uint64_t>(offset),
                            static_cast<uint64_t>(size),
                            res, status, err)) {
            send_error(res, status, err);
            return;
        }
    });
}

} // namespace

int main(int argc, char** argv) {
    std::string host = kDefaultHost;
    int port = kDefaultPort;
    std::string index_cache_dir = kDefaultIndexCacheDir;

    CLI::App app{"T-BLAEQ JSON index HTTP server"};
    app.add_option("-p,--port", port, "HTTP listen port (default: 8090)");
    app.add_option("-i,--index-cache", index_cache_dir, "Index cache directory (default: ./tempJsonIndexCache)");
    app.add_option("-H,--host", host, "HTTP listen host (default: 0.0.0.0)");
    CLI11_PARSE(app, argc, argv);

    if (port <= 0 || port > 65535) {
        throw std::runtime_error("Invalid port. Must be in range (0, 65535].");
    }

    fs::create_directories(index_cache_dir);

    JsonIndexManager manager(index_cache_dir);
    httplib::Server server;
    setup_routes(server, manager);

    std::cout << "T-BLAEQ JSON index server listening on " << host << ":" << port
              << ", index cache path: " << index_cache_dir << "\n";

    if (!server.listen(host, port)) {
        throw std::runtime_error("Failed to start HTTP server.");
    }
    return 0;
}
