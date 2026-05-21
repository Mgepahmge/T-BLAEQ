#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include "CLI11.hpp"
#include "src/Data_Structures/File.cuh"
#include "src/core/Memory.cuh"
#include "src/core/QueryHandler.cuh"

static std::vector<size_t> parseRatiosCsv(const std::string& s) {
    std::vector<size_t> ratios;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            throw std::invalid_argument("Invalid --ratios: empty segment in '" + s + "'");
        }
        const size_t r = static_cast<size_t>(std::stoull(token));
        if (r == 0) {
            throw std::invalid_argument("Invalid --ratios: ratio must be > 0");
        }
        ratios.push_back(r);
    }
    if (ratios.empty()) {
        throw std::invalid_argument("Invalid --ratios: no ratios parsed from '" + s + "'");
    }
    return ratios;
}

static bool isClose(double actual, double expected, double absTol, double relTol, double& absDiff, double& relDiff) {
    absDiff = std::abs(actual - expected);
    relDiff = absDiff / std::max(1.0, std::abs(expected));
    return absDiff <= absTol || relDiff <= relTol;
}

int main(int argc, char** argv) {
    CLI::App app{"T-BLAEQ: layout consistency check"};

    std::string datasetPath;
    std::string queryFilePath;
    app.add_option("-d,--dataset", datasetPath, "Path to dataset file")->required();
    app.add_option("-f,--query-file", queryFilePath, "Path to query file")->required();

    int queryTypeInt = 0;
    app.add_option("-t,--query-type", queryTypeInt, "Query type: 0 = Range, 1 = KNN (default: 0)");

    int k = 10;
    app.add_option("-k,--knn-k", k, "K for KNN queries (default: 10)");

    int maxQueryCount = 1;
    app.add_option("-q,--max-queries", maxQueryCount, "Maximum number of queries to run (default: 1)");

    bool forceUseCPU = false;
    app.add_option("--force-cpu", forceUseCPU, "Force CPU index building (default: false)");

    size_t buildHeight = 4;
    std::string buildRatiosStr = "100,50,20";
    app.add_option("--height", buildHeight, "Hierarchy height (levels), must be >= 2 (default: 4)");
    app.add_option("--ratios", buildRatiosStr,
                   "Comma-separated coarsening ratios, count must equal height-1 (default: 100,50,20)");

    double absTol = 1e-6;
    double relTol = 1e-6;
    app.add_option("--abs-tol", absTol, "Absolute tolerance for vector comparison (default: 1e-6)");
    app.add_option("--rel-tol", relTol, "Relative tolerance for vector comparison (default: 1e-6)");

    CLI11_PARSE(app, argc, argv);

    if (buildHeight < 2) {
        throw std::runtime_error("--height must be >= 2");
    }
    const std::vector<size_t> buildRatios = parseRatiosCsv(buildRatiosStr);
    if (buildRatios.size() != buildHeight - 1) {
        throw std::runtime_error("--ratios count must equal --height - 1");
    }
    if (queryTypeInt != 0 && queryTypeInt != 1) {
        throw std::runtime_error("Unsupported query type: " + std::to_string(queryTypeInt));
    }
    if (maxQueryCount <= 0) {
        throw std::runtime_error("--max-queries must be >= 1");
    }

    const PointCloud dataset = loadFromFile(datasetPath);
    if (dataset.size <= 0 || dataset.dim <= 0) {
        throw std::runtime_error("Dataset is empty or invalid");
    }

    QueryHandler handler(forceUseCPU, datasetPath, buildHeight, buildRatios);
    const QueryType qType = (queryTypeInt == 1) ? QueryType::POINT : QueryType::RANGE;
    QueryResult result = handler.performQuery(queryFilePath, qType, /*saveFineMesh=*/true,
                                              maxQueryCount, static_cast<size_t>(k));

    if (result.errorCode != 0) {
        throw std::runtime_error("Query failed with errorCode=" + std::to_string(result.errorCode));
    }
    if (result.fineMesh.empty()) {
        throw std::runtime_error("Query returned no fine-mesh results to validate");
    }

    SparseGrid* grid = result.fineMesh.front();
    const size_t nnz = grid->get_nnz_nums();
    if (nnz == 0) {
        throw std::runtime_error("Query returned empty result set");
    }

    const size_t dim = grid->get_dimensions();
    if (dim != static_cast<size_t>(dataset.dim)) {
        throw std::runtime_error("Dataset dimensionality mismatch with query result");
    }

    std::vector<size_t> ids(nnz);
    std::vector<double> vals(nnz * dim);

    const bool onHost = !result.fineMeshOnHost.empty() && result.fineMeshOnHost.front();
    if (onHost) {
        std::copy(grid->get_ids_(), grid->get_ids_() + nnz, ids.begin());
        std::copy(grid->get_vals_(), grid->get_vals_() + nnz * dim, vals.begin());
    } else {
        HostBuffer<size_t> hIds(nnz);
        hIds.downloadFrom(grid->get_ids_(), nnz);
        std::copy(hIds.data(), hIds.data() + nnz, ids.begin());

        HostBuffer<double> hVals(nnz * dim);
        hVals.downloadFrom(grid->get_vals_(), nnz * dim);
        std::copy(hVals.data(), hVals.data() + nnz * dim, vals.begin());
    }

    size_t badVectors = 0;
    size_t badValues = 0;
    double maxAbs = 0.0;
    double maxRel = 0.0;
    size_t maxVecIdx = 0;
    size_t maxDimIdx = 0;
    size_t firstBadVec = std::numeric_limits<size_t>::max();
    size_t firstBadDim = 0;
    double firstExpected = 0.0;
    double firstActual = 0.0;

    for (size_t i = 0; i < nnz; ++i) {
        const size_t id = ids[i];
        if (id >= static_cast<size_t>(dataset.size)) {
            ++badVectors;
            if (firstBadVec == std::numeric_limits<size_t>::max()) {
                firstBadVec = i;
                firstBadDim = 0;
                firstExpected = 0.0;
                firstActual = 0.0;
            }
            continue;
        }

        bool vectorOk = true;
        const double* expected = dataset.pointAt(static_cast<int>(id));
        for (size_t d = 0; d < dim; ++d) {
            double absDiff = 0.0;
            double relDiff = 0.0;
            const double actual = vals[i * dim + d];
            const double expVal = expected[d];
            const bool ok = isClose(actual, expVal, absTol, relTol, absDiff, relDiff);

            if (absDiff > maxAbs) {
                maxAbs = absDiff;
                maxRel = relDiff;
                maxVecIdx = i;
                maxDimIdx = d;
            }

            if (!ok) {
                ++badValues;
                vectorOk = false;
                if (firstBadVec == std::numeric_limits<size_t>::max()) {
                    firstBadVec = i;
                    firstBadDim = d;
                    firstExpected = expVal;
                    firstActual = actual;
                }
            }
        }
        if (!vectorOk) {
            ++badVectors;
        }
    }

    if (badVectors == 0) {
        std::cout << "Layout consistency check passed (" << nnz << " vectors, D=" << dim
                  << "). Max abs diff=" << maxAbs << ", max rel diff=" << maxRel << "\n";
        return 0;
    }

    std::cerr << "Layout consistency check failed: " << badVectors << "/" << nnz
              << " vectors mismatch (" << badValues << " values).\n";
    if (firstBadVec != std::numeric_limits<size_t>::max()) {
        std::cerr << "First mismatch at result[" << firstBadVec << "] id=" << ids[firstBadVec]
                  << " dim=" << firstBadDim << " expected=" << firstExpected
                  << " actual=" << firstActual << " (absTol=" << absTol
                  << ", relTol=" << relTol << ")\n";
    }
    std::cerr << "Max abs diff at result[" << maxVecIdx << "] dim=" << maxDimIdx
              << " = " << maxAbs << " (rel " << maxRel << ")\n";
    return 1;
}
