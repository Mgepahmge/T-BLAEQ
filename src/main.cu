#include <iostream>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>
#include "CLI11.hpp"
#include "src/core/QueryHandler.cuh"
#include "src/core/QueryEngine.cuh"    // saveQueryResult
#include "src/core/MemoryPolicy.cuh"   // IndexPolicy::parseLevel

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

int main(int argc, char** argv) {

    CLI::App app{"T-BLAEQ: Index Builder and Query Engine"};

    // Mode
    bool buildFlag = false;
    bool queryFlag = false;
    bool genFlag   = false;
    auto* modeGroup = app.add_option_group("mode");
    modeGroup->add_flag("--build-index", buildFlag,
                        "Build and save index from dataset");
    modeGroup->add_flag("--test-query",  queryFlag,
                        "Run queries against a saved index");
    modeGroup->add_flag("--gen-index",   genFlag,
                        "Generate a random synthetic index and save it");
    modeGroup->require_option(1, 1);

    // Paths
    std::string datasetPath;
    app.add_option("-d,--dataset",    datasetPath,  "Path to dataset file");

    std::string indexPath = "indexes/";
    app.add_option("-i,--index-path", indexPath,
                   "Path to save / load index (default: indexes/)");

    std::string queryFilePath;
    app.add_option("-f,--query-file", queryFilePath, "Path to query file");

    std::string resultPath = "tempResult.csv";
    app.add_option("-r,--result-path", resultPath,
                   "Path to save results CSV (default: tempResult.csv)");

    // Query parameters
    int maxQueryCount = 10;
    app.add_option("-q,--max-queries", maxQueryCount,
                   "Maximum number of queries to run (default: 10)");

    int queryTypeInt = 0;
    app.add_option("-t,--query-type", queryTypeInt,
                   "Query type: 0 = Range, 1 = KNN (default: 0)");

    int k = 10;
    app.add_option("-k,--knn-k", k,
                   "K for KNN queries (default: 10)");

    bool forceUseCPU = false;
    app.add_option("--force-cpu", forceUseCPU, "Force CPU index building (default : false)");

    // Index hierarchy parameters (apply to both --build-index and --gen-index)
    size_t buildHeight = 4;
    std::string buildRatiosStr = "100,50,20";
    app.add_option("--height", buildHeight, "Hierarchy height (levels), must be >= 2 (default: 4)");
    app.add_option("--ratios", buildRatiosStr,
                   "Comma-separated coarsening ratios, count must equal height-1 (default: 100,50,20)");

    // Random index generation parameters
    size_t genN = 1000000;
    size_t genD = 3;
    double genMin = 1.0;
    double genMax = 100.0;
    bool   genInt = false;
    uint64_t genSeed = 12345;
    double genSigmaDivisor = 3.0;
    app.add_option("--gen-N",   genN,   "Number of points at the finest mesh level (default: 1000000)");
    app.add_option("--gen-D",   genD,   "Dimensionality of each point (default: 3)");
    app.add_option("--gen-min", genMin, "Lower bound of the value range, must be > 0 (default: 1.0)");
    app.add_option("--gen-max", genMax, "Upper bound of the value range (default: 100.0)");
    app.add_flag  ("--gen-int", genInt, "Generate integer coordinates (default: false)");
    app.add_option("--gen-seed", genSeed, "Random seed for RandomKmeans hierarchy generation (default: 12345)");
    app.add_option("--gen-sigma-divisor", genSigmaDivisor,
                   "Random spread control: sigma = spacing/divisor, must be > 0 (default: 3.0)");

    // Policy override
    std::string forcePolicyStr;
    app.add_option("--force-policy", forcePolicyStr,
                   "Force all hierarchy levels to use a specific memory policy.\n"
                   "Valid values: L0, L1, L2, L3.\n"
                   "If omitted, the scheduler auto-selects per-level policies\n"
                   "based on available GPU memory.");

    CLI11_PARSE(app, argc, argv);

    if (buildHeight < 2) {
        throw std::runtime_error("--height must be >= 2");
    }
    const std::vector<size_t> buildRatios = parseRatiosCsv(buildRatiosStr);
    if (buildRatios.size() != buildHeight - 1) {
        throw std::runtime_error("--ratios count must equal --height - 1");
    }

    // Build
    if (buildFlag) {
        QueryHandler handler(forceUseCPU, datasetPath, buildHeight, buildRatios);
        handler.saveIndex(indexPath);
    }

    // Generate random index
    if (genFlag) {
        if (genSigmaDivisor <= 0.0) {
            throw std::runtime_error("--gen-sigma-divisor must be > 0");
        }

        QueryHandler handler(genN, genD, genMin, genMax, genInt, buildHeight, buildRatios, genSeed, genSigmaDivisor);
        handler.saveIndex(indexPath);
    }

    // Query
    if (queryFlag) {
        QueryHandler handler(indexPath, /*loadFromIndex=*/true);

        if (queryTypeInt != 0 && queryTypeInt != 1)
            throw std::runtime_error(
                "Unsupported query type: " + std::to_string(queryTypeInt));
        const QueryType qType = (queryTypeInt == 1)
                              ? QueryType::POINT : QueryType::RANGE;

        // Apply policy: manual override or auto-select
        if (!forcePolicyStr.empty()) {
            const LevelPolicy lp = IndexPolicy::parseLevel(forcePolicyStr);
            std::cout << "Policy override: all levels forced to "
                      << levelPolicyName(lp) << "\n";
            handler.prepareForQuery(lp);
        } else {
            handler.prepareForQuery();
        }

        QueryResult result = handler.performQuery(
            queryFilePath, qType,
            /*saveFineMesh=*/false,
            maxQueryCount,
            static_cast<size_t>(k));

        saveQueryResult(result, resultPath);
        std::cout << "Results saved to: " << resultPath << "\n";
    }

    return 0;
}
