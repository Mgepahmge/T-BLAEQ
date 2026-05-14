/**
 * @file HTTPServer.cu
 * @brief HTTP+JSON service layer for index building and online query execution.
 *
 * @details This executable exposes T-BLAEQ over HTTP using cpp-httplib and picojson.
 * It provides three endpoints:
 *
 * 1) POST /build-index
 *    Input JSON:
 *      - dataset_name (string, required)
 *      - dataset_path (string, required when index does not exist; may be omitted only
 *        when the named index already exists)
 *      - force_cpu (bool, optional, default false)
 *
 *    Output JSON (success):
 *      - success (true)
 *      - dataset_name (string)
 *      - already_exists (bool)
 *      - message (string)
 *
 *    Output JSON (failure):
 *      - success (false)
 *      - message (string)
 *
 * 2) GET/POST /datasets
 *    Input: none
 *    Output JSON (success):
 *      - success (true)
 *      - datasets_count (number): length of datasets array
 *      - datasets (array<string>): names of all datasets whose index directory
 *        contains metadata.bin
 *
 * 3) POST /query
 *    Input JSON:
 *      - dataset_name (string, required)
 *      - query_type (string, required: "KNN" or "RANGE", case-insensitive)
 *      - KNN mode:
 *          - query_point (array<number>, required, length == dataset dimension)
 *          - k (positive integer, optional, default 10)
 *      - RANGE mode:
 *          - upper_bound (array<number>, required, length == dataset dimension)
 *          - lower_bound (array<number>, required, length == dataset dimension)
 *          - requires lower_bound[i] <= upper_bound[i] for all i
 *
 *    Output JSON (success):
 *      - success (true)
 *      - dataset_name (string)
 *      - query_type (string)
 *      - result_count (number): length of result array
 *      - result (array<number>): fine-mesh point ids (extractResult output)
 *      - level_original_sizes (array<number>): each runLevel target layer's original size
 *      - level_pruned_sizes (array<number>): fineGrid nnz after each runLevel for this query
 *      - query_runtime (object): query_time_us/query_time_ms/total_time_us/avg_time_us/query_count
 *
 *    Output JSON (failure):
 *      - success (false)
 *      - message (string)
 *
 * Validation policy:
 *   - All request bodies must be JSON objects.
 *   - Required fields must exist and have correct types.
 *   - Dataset names are restricted to [A-Za-z0-9._-].
 *   - Vector dimensions are validated against index dimensionality.
 */
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>
#include "CLI11.hpp"
#include "core/QueryHandler.cuh"
#include "cuda_runtime.h"
#include "httplib.h"
#include "picojson.h"

namespace fs = std::filesystem;

namespace {

constexpr const char* kDefaultHost = "0.0.0.0";
constexpr int kDefaultPort = 8080;
constexpr const char* kDefaultIndexCacheDir = "./tempIndexCache";

std::string processIndexPath(const std::string& datasetName, const std::string& basePath) {
    return (fs::path(basePath) / datasetName).string();
}

bool isValidDatasetName(const std::string& datasetName) {
    static const std::regex kDatasetNamePattern("^[A-Za-z0-9._-]+$");
    return !datasetName.empty() && std::regex_match(datasetName, kDatasetNamePattern);
}

bool indexExists(const std::string& datasetName, const std::string& basePath) {
    const fs::path indexDir = processIndexPath(datasetName, basePath);
    return fs::exists(indexDir / "metadata.bin");
}

std::vector<std::string> listIndexedDatasets(const std::string& basePath) {
    std::vector<std::string> datasets;
    const fs::path base(basePath);
    if (!fs::exists(base) || !fs::is_directory(base)) {
        return datasets;
    }

    for (const auto& entry : fs::directory_iterator(base)) {
        if (!entry.is_directory()) {
            continue;
        }
        const fs::path metadata = entry.path() / "metadata.bin";
        if (fs::exists(metadata)) {
            datasets.push_back(entry.path().filename().string());
        }
    }
    std::sort(datasets.begin(), datasets.end());
    return datasets;
}

void setJsonResponse(httplib::Response& res, const picojson::object& body, int statusCode = 200) {
    res.status = statusCode;
    res.set_header("Content-Type", "application/json");
    res.set_content(picojson::value(body).serialize(), "application/json");
}

void sendError(httplib::Response& res, int statusCode, const std::string& message) {
    picojson::object body;
    body["success"] = picojson::value(false);
    body["message"] = picojson::value(message);
    setJsonResponse(res, body, statusCode);
}

bool parseRequestJson(const httplib::Request& req, httplib::Response& res, picojson::object& out) {
    picojson::value parsed;
    const std::string err = picojson::parse(parsed, req.body);
    if (!err.empty()) {
        sendError(res, 400, "Invalid JSON: " + err);
        return false;
    }
    if (!parsed.is<picojson::object>()) {
        sendError(res, 400, "Request body must be a JSON object");
        return false;
    }
    out = parsed.get<picojson::object>();
    return true;
}

bool requireString(const picojson::object& obj, const std::string& field, std::string& out) {
    const auto it = obj.find(field);
    if (it == obj.end() || !it->second.is<std::string>()) {
        return false;
    }
    out = it->second.get<std::string>();
    return true;
}

bool optionalBool(const picojson::object& obj, const std::string& field, bool& out) {
    const auto it = obj.find(field);
    if (it == obj.end()) {
        return true;
    }
    if (!it->second.is<bool>()) {
        return false;
    }
    out = it->second.get<bool>();
    return true;
}

bool optionalPositiveInteger(const picojson::object& obj, const std::string& field, size_t& out) {
    const auto it = obj.find(field);
    if (it == obj.end()) {
        return true;
    }
    if (!it->second.is<double>()) {
        return false;
    }
    const double value = it->second.get<double>();
    if (value <= 0 || std::floor(value) != value) {
        return false;
    }
    out = static_cast<size_t>(value);
    return true;
}

bool parseNumericVector(const picojson::value& value, std::vector<double>& out) {
    if (!value.is<picojson::array>()) {
        return false;
    }
    const auto& arr = value.get<picojson::array>();
    out.clear();
    out.reserve(arr.size());
    for (const auto& elem : arr) {
        if (!elem.is<double>()) {
            return false;
        }
        out.push_back(elem.get<double>());
    }
    return true;
}

picojson::array toJsonArray(const std::vector<size_t>& values) {
    picojson::array json;
    json.reserve(values.size());
    for (const size_t v : values) {
        json.emplace_back(static_cast<double>(v));
    }
    return json;
}

std::vector<size_t> extractResult(const QueryResult& queryResult) {
    if (queryResult.queryCount != 1 || queryResult.fineMesh.empty() || queryResult.fineMeshOnHost.empty()) {
        throw std::runtime_error("Query result is invalid. Ensure exactly one query and saveFineMesh=true.");
    }
    const auto& fineMesh = *queryResult.fineMesh[0];
    const size_t size = fineMesh.get_nnz_nums();
    std::vector<size_t> result(size);
    if (!fineMesh.get_ids_()) {
        return result;
    }
    if (queryResult.fineMeshOnHost[0]) {
        std::copy_n(fineMesh.get_ids_(), size, result.begin());
        return result;
    }
    const cudaError_t err = cudaMemcpy(result.data(), fineMesh.get_ids_(), size * sizeof(size_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpy failed: ") + cudaGetErrorString(err));
    }
    return result;
}

class IndexManager {
public:
    /*!
     * @brief Create an index manager rooted at the given cache directory.
     * @param basePath Root directory that stores one index subdirectory per dataset.
     */
    explicit IndexManager(std::string basePath) : basePath_(std::move(basePath)) {}

    /*!
     * @brief Check whether a dataset index is already persisted on disk.
     * @param datasetName Dataset logical name.
     * @return True when metadata.bin exists under the dataset index directory.
     */
    bool existsOnDisk(const std::string& datasetName) const {
        return indexExists(datasetName, basePath_);
    }

    /*!
     * @brief List all dataset names that currently have persisted indexes.
     * @return Sorted list of dataset names.
     */
    std::vector<std::string> listDatasets() const {
        return listIndexedDatasets(basePath_);
    }

    /*!
     * @brief Build an index from dataset file and persist it under datasetName.
     * @param datasetName Target dataset name (directory name under cache root).
     * @param datasetPath Input dataset file path.
     * @param forceCPU Whether to force CPU index building.
     */
    void buildAndSave(const std::string& datasetName, const std::string& datasetPath, bool forceCPU) {
        auto handler = std::make_unique<QueryHandler>(forceCPU, datasetPath);
        handler->saveIndex(processIndexPath(datasetName, basePath_));

        std::lock_guard<std::mutex> lock(mutex_);
        handlers_[datasetName] = std::move(handler);
    }

    QueryHandler& getOrLoad(const std::string& datasetName) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = handlers_.find(datasetName);
        if (it != handlers_.end()) {
            return *it->second;
        }

        if (!existsOnDisk(datasetName)) {
            throw std::runtime_error("Dataset index does not exist: " + datasetName);
        }
        auto handler = std::make_unique<QueryHandler>(processIndexPath(datasetName, basePath_), true);
        handler->prepareForQuery();
        QueryHandler& ref = *handler;
        handlers_[datasetName] = std::move(handler);
        return ref;
    }

private:
    std::string basePath_;
    std::unordered_map<std::string, std::unique_ptr<QueryHandler>> handlers_;
    mutable std::mutex mutex_;
};

/*!
 * @brief Register all HTTP routes for index build/list/query services.
 *
 * @details Endpoint I/O contract:
 *
 * - POST /build-index
 *   Request:
 *   @code{.json}
 *   {"dataset_name":"sift","dataset_path":"/data/sift.txt","force_cpu":false}
 *   @endcode
 *   Success response:
 *   @code{.json}
 *   {"success":true,"dataset_name":"sift","already_exists":false,"message":"Index build and save succeeded"}
 *   @endcode
 *
 * - GET /datasets (or POST /datasets)
 *   Success response:
 *   @code{.json}
 *   {"success":true,"datasets_count":2,"datasets":["sift","gist"]}
 *   @endcode
 *
 * - POST /query
 *   KNN request:
 *   @code{.json}
 *   {"dataset_name":"sift","query_type":"KNN","query_point":[1.0,2.0,3.0],"k":10}
 *   @endcode
 *   RANGE request:
 *   @code{.json}
 *   {"dataset_name":"sift","query_type":"RANGE","lower_bound":[0.0,0.0,0.0],"upper_bound":[5.0,5.0,5.0]}
 *   @endcode
 *   Success response:
 *   @code{.json}
 *   {"success":true,"dataset_name":"sift","query_type":"KNN","result_count":3,"result":[12,45,98],
 *    "level_original_sizes":[1000,20000,1000000],"level_pruned_sizes":[97,1043,3],
 *    "query_runtime":{"query_time_us":8123,"query_time_ms":8.123,"total_time_us":8123,"avg_time_us":8123.0,"query_count":1}}
 *   @endcode
 *
 * All failures use:
 * @code{.json}
 * {"success":false,"message":"..."}
 * @endcode
 *
 * @param server cpp-httplib server instance.
 * @param manager Index manager used by route handlers.
 */
void setupRoutes(httplib::Server& server, IndexManager& manager) {
    server.Post("/build-index", [&manager](const httplib::Request& req, httplib::Response& res) {
        picojson::object body;
        if (!parseRequestJson(req, res, body)) {
            return;
        }

        std::string datasetName;
        if (!requireString(body, "dataset_name", datasetName)) {
            sendError(res, 400, "Field 'dataset_name' is required and must be a string");
            return;
        }
        if (!isValidDatasetName(datasetName)) {
            sendError(res, 400, "Invalid dataset_name. Allowed: [A-Za-z0-9._-], non-empty");
            return;
        }

        if (manager.existsOnDisk(datasetName)) {
            picojson::object ok;
            ok["success"] = picojson::value(true);
            ok["dataset_name"] = picojson::value(datasetName);
            ok["already_exists"] = picojson::value(true);
            ok["message"] = picojson::value("Index already exists");
            setJsonResponse(res, ok);
            return;
        }

        std::string datasetPath;
        if (!requireString(body, "dataset_path", datasetPath) || datasetPath.empty()) {
            sendError(res, 400, "Field 'dataset_path' is required and cannot be empty when index does not exist");
            return;
        }
        if (!fs::exists(datasetPath) || !fs::is_regular_file(datasetPath)) {
            sendError(res, 400, "dataset_path does not exist or is not a file");
            return;
        }

        bool forceCPU = false;
        if (!optionalBool(body, "force_cpu", forceCPU)) {
            sendError(res, 400, "Field 'force_cpu' must be a boolean");
            return;
        }

        try {
            manager.buildAndSave(datasetName, datasetPath, forceCPU);
            picojson::object ok;
            ok["success"] = picojson::value(true);
            ok["dataset_name"] = picojson::value(datasetName);
            ok["already_exists"] = picojson::value(false);
            ok["message"] = picojson::value("Index build and save succeeded");
            setJsonResponse(res, ok);
        } catch (const std::exception& e) {
            sendError(res, 500, std::string("Failed to build index: ") + e.what());
        }
    });

    auto listHandler = [&manager](const httplib::Request&, httplib::Response& res) {
        const std::vector<std::string> datasets = manager.listDatasets();
        picojson::array datasetsJson;
        datasetsJson.reserve(datasets.size());
        for (const auto& name : datasets) {
            datasetsJson.emplace_back(name);
        }
        picojson::object body;
        body["success"] = picojson::value(true);
        body["datasets_count"] = picojson::value(static_cast<double>(datasets.size()));
        body["datasets"] = picojson::value(datasetsJson);
        setJsonResponse(res, body);
    };
    server.Get("/datasets", listHandler);
    server.Post("/datasets", listHandler);

    server.Post("/query", [&manager](const httplib::Request& req, httplib::Response& res) {
        picojson::object body;
        if (!parseRequestJson(req, res, body)) {
            return;
        }

        std::string datasetName;
        if (!requireString(body, "dataset_name", datasetName)) {
            sendError(res, 400, "Field 'dataset_name' is required and must be a string");
            return;
        }
        if (!isValidDatasetName(datasetName)) {
            sendError(res, 400, "Invalid dataset_name. Allowed: [A-Za-z0-9._-], non-empty");
            return;
        }

        std::string queryTypeStr;
        if (!requireString(body, "query_type", queryTypeStr)) {
            sendError(res, 400, "Field 'query_type' is required and must be 'KNN' or 'RANGE'");
            return;
        }
        for (char& c : queryTypeStr) {
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        }
        if (queryTypeStr != "KNN" && queryTypeStr != "RANGE") {
            sendError(res, 400, "query_type must be 'KNN' or 'RANGE'");
            return;
        }

        QueryHandler* handler = nullptr;
        try {
            handler = &manager.getOrLoad(datasetName);
        } catch (const std::exception& e) {
            sendError(res, 400, e.what());
            return;
        }

        const size_t dim = handler->getDim();
        std::vector<size_t> resultIds;
        std::vector<size_t> levelOriginalSizes;
        std::vector<size_t> levelPrunedSizes;
        std::string memoryPolicy;
        long queryTimeUs = 0;
        long totalTimeUs = 0;
        double avgTimeUs = 0.0;
        size_t queryCount = 0;

        try {
            auto collectQueryResult = [&](const QueryResult& qr) {
                resultIds = extractResult(qr);
                levelOriginalSizes = qr.levelOriginalSize;
                if (!qr.levelFineMeshSize.empty()) {
                    levelPrunedSizes = qr.levelFineMeshSize[0];
                }
                if (!qr.queryTimeUs.empty()) {
                    queryTimeUs = qr.queryTimeUs[0];
                }
                totalTimeUs = qr.totalTimeUs;
                queryCount = qr.queryCount;
                avgTimeUs = (queryCount > 0) ? static_cast<double>(totalTimeUs) / static_cast<double>(queryCount) : 0.0;
                memoryPolicy = qr.memoryPolicy;
            };

            if (queryTypeStr == "KNN") {
                const auto it = body.find("query_point");
                if (it == body.end()) {
                    sendError(res, 400, "Field 'query_point' is required for KNN query");
                    return;
                }
                std::vector<double> point;
                if (!parseNumericVector(it->second, point)) {
                    sendError(res, 400, "Field 'query_point' must be a numeric array");
                    return;
                }
                if (point.size() != dim) {
                    sendError(res, 400, "query_point dimension mismatch: expected " + std::to_string(dim));
                    return;
                }

                size_t k = 10;
                if (!optionalPositiveInteger(body, "k", k)) {
                    sendError(res, 400, "Field 'k' must be a positive integer");
                    return;
                }
                const QueryResult qr = handler->performSingleKNNQuery(point, k, true);
                collectQueryResult(qr);
            } else {
                const auto upperIt = body.find("upper_bound");
                const auto lowerIt = body.find("lower_bound");
                if (upperIt == body.end() || lowerIt == body.end()) {
                    sendError(res, 400, "Fields 'upper_bound' and 'lower_bound' are required for RANGE query");
                    return;
                }

                std::vector<double> upperBound;
                std::vector<double> lowerBound;
                if (!parseNumericVector(upperIt->second, upperBound) || !parseNumericVector(lowerIt->second, lowerBound)) {
                    sendError(res, 400, "upper_bound and lower_bound must be numeric arrays");
                    return;
                }
                if (upperBound.size() != dim || lowerBound.size() != dim) {
                    sendError(res, 400, "Range bound dimension mismatch: expected " + std::to_string(dim));
                    return;
                }
                for (size_t i = 0; i < dim; ++i) {
                    if (lowerBound[i] > upperBound[i]) {
                        sendError(res, 400, "Invalid range: lower_bound[" + std::to_string(i) + "] > upper_bound[" + std::to_string(i) + "]");
                        return;
                    }
                }

                const QueryResult qr = handler->performSingleRangeQuery(upperBound, lowerBound, true);
                collectQueryResult(qr);
            }
        } catch (const std::exception& e) {
            sendError(res, 500, std::string("Query execution failed: ") + e.what());
            return;
        }

        const picojson::array resultJson = toJsonArray(resultIds);

        picojson::object runtimeJson;
        runtimeJson["query_time_us"] = picojson::value(static_cast<double>(queryTimeUs));
        runtimeJson["query_time_ms"] = picojson::value(static_cast<double>(queryTimeUs) / 1000.0);
        runtimeJson["total_time_us"] = picojson::value(static_cast<double>(totalTimeUs));
        runtimeJson["avg_time_us"] = picojson::value(avgTimeUs);
        runtimeJson["query_count"] = picojson::value(static_cast<double>(queryCount));

        picojson::object ok;
        ok["success"] = picojson::value(true);
        ok["dataset_name"] = picojson::value(datasetName);
        ok["query_type"] = picojson::value(queryTypeStr);
        ok["result_count"] = picojson::value(static_cast<double>(resultIds.size()));
        ok["result"] = picojson::value(resultJson);
        ok["memory_policy"] = picojson::value(memoryPolicy);
        ok["level_original_sizes"] = picojson::value(toJsonArray(levelOriginalSizes));
        ok["level_pruned_sizes"] = picojson::value(toJsonArray(levelPrunedSizes));
        ok["query_runtime"] = picojson::value(runtimeJson);
        setJsonResponse(res, ok);
    });
}

} // namespace

/*!
 * @brief Entry point of the HTTP server executable.
 *
 * @details Command line options:
 * - --port, -p: listen port (default: 8080)
 * - --index-cache, -i: index cache root path (default: ./tempIndexCache)
 * - --host, -H: bind host (default: 0.0.0.0)
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return 0 on normal shutdown.
 */
int main(int argc, char** argv) {
    std::string host = kDefaultHost;
    int port = kDefaultPort;
    std::string indexCacheDir = kDefaultIndexCacheDir;

    CLI::App app{"T-BLAEQ HTTP server"};
    app.add_option("-p,--port", port, "HTTP listen port (default: 8080)");
    app.add_option("-i,--index-cache", indexCacheDir, "Index cache directory (default: ./tempIndexCache)");
    app.add_option("-H,--host", host, "HTTP listen host (default: 0.0.0.0)");
    CLI11_PARSE(app, argc, argv);

    if (port <= 0 || port > 65535) {
        throw std::runtime_error("Invalid port. Must be in range (0, 65535].");
    }

    fs::create_directories(indexCacheDir);

    IndexManager manager(indexCacheDir);
    httplib::Server server;
    setupRoutes(server, manager);

    std::cout << "T-BLAEQ HTTP server listening on " << host << ":" << port
              << ", index cache path: " << indexCacheDir << "\n";

    if (!server.listen(host, port)) {
        throw std::runtime_error("Failed to start HTTP server.");
    }
    return 0;
}
