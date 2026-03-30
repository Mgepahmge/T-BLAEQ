#include <iostream>
#include <filesystem>
#include <string>
#include "CLI11.hpp"
#include "src/core/QueryHandler.cuh"
#include "src/core/QueryEngine.cuh"    // saveQueryResult
#include "src/core/MemoryPolicy.cuh"   // IndexPolicy::parseLevel

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

    // Random index generation parameters
    size_t genN = 1000000;
    size_t genD = 3;
    double genMin = 1.0;
    double genMax = 100.0;
    bool   genInt = false;
    app.add_option("--gen-N",   genN,   "Number of points at the finest mesh level (default: 1000000)");
    app.add_option("--gen-D",   genD,   "Dimensionality of each point (default: 3)");
    app.add_option("--gen-min", genMin, "Lower bound of the value range, must be > 0 (default: 1.0)");
    app.add_option("--gen-max", genMax, "Upper bound of the value range (default: 100.0)");
    app.add_flag  ("--gen-int", genInt, "Generate integer coordinates (default: false)");

    // Policy override
    std::string forcePolicyStr;
    app.add_option("--force-policy", forcePolicyStr,
                   "Force all hierarchy levels to use a specific memory policy.\n"
                   "Valid values: L0, L1, L2, L3.\n"
                   "If omitted, the scheduler auto-selects per-level policies\n"
                   "based on available GPU memory.");

    CLI11_PARSE(app, argc, argv);

    // Build
    if (buildFlag) {
        QueryHandler handler(forceUseCPU, datasetPath);
        handler.saveIndex(indexPath);
    }

    // Generate random index
    if (genFlag) {
        QueryHandler handler(genN, genD, genMin, genMax, genInt);
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
