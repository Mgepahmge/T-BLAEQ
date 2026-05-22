#include <iostream>
#include <stdexcept>
#include "QueryHandler.cuh"
#include "src/Data_Structures/File.cuh"
#include "src/func.hpp"
#include "src/utils/Utils.cuh"

namespace {

struct BuildPlan {
    size_t height = 0;
    std::vector<size_t> ratios;
    bool truncated = false;
};

size_t computeCentroidCountSim(size_t dataNums, size_t ratio) {
    if (ratio == 0 || dataNums / ratio == 0) {
        return 1;
    }
    return dataNums / ratio;
}

BuildPlan adjustBuildPlan(size_t N, size_t height, const std::vector<size_t>& ratios) {
    if (height < 2) {
        throw std::invalid_argument("QueryHandler: height must be >= 2");
    }
    if (ratios.size() != height - 1) {
        throw std::invalid_argument("QueryHandler: ratios size must equal height - 1");
    }
    size_t dataNums = N;
    size_t keep = ratios.size();
    for (size_t i = 0; i < ratios.size(); ++i) {
        const size_t centroidNums = computeCentroidCountSim(dataNums, ratios[i]);
        if (centroidNums == 1) {
            keep = i + 1;
            break;
        }
        dataNums = centroidNums;
    }
    BuildPlan plan;
    plan.height = keep + 1;
    plan.ratios.assign(ratios.begin(), ratios.begin() + static_cast<long>(keep));
    plan.truncated = (keep != ratios.size());
    return plan;
}

void printBuildConfig(const BuildPlan& plan) {
    std::cout << "Build config: height=" << plan.height << " ratios=[";
    for (size_t i = 0; i < plan.ratios.size(); ++i) {
        if (i) {
            std::cout << ",";
        }
        std::cout << plan.ratios[i];
    }
    std::cout << "]\n";
}

} // namespace

QueryHandler::QueryHandler(const bool forceUseCPU, const std::string& datasetPath,
                           const size_t height, const std::vector<size_t>& ratios) {
    std::cout << "T-BLAEQ: building index from " << datasetPath << "\n";
    const std::string name = extractDatasetName(datasetPath);
    const PointCloud dataset = loadFromFile(datasetPath);
    const BuildPlan plan = adjustBuildPlan(static_cast<size_t>(dataset.size), height, ratios);
    if (plan.truncated) {
        std::cout << "Truncating build plan to avoid degenerate top layer.\n";
    }
    printBuildConfig(plan);

    const auto t0 = std::chrono::steady_clock::now();
    idx_.reset(IndexBuilder::build(dataset.data, static_cast<size_t>(dataset.size), static_cast<size_t>(dataset.dim),
                                   name, forceUseCPU, plan.height, plan.ratios));
    idx_->datasetName = extractDatasetName(datasetPath);
    const auto t1 = std::chrono::steady_clock::now();
    Chrono::printElapsed("Index build total", t0, t1);
}

QueryHandler::QueryHandler(const bool forceUseCPU, const PointCloud& dataset, const std::string& name,
                           const size_t height, const std::vector<size_t>& ratios) {
    if (dataset.data == nullptr || dataset.size <= 0 || dataset.dim <= 0) {
        throw std::invalid_argument("QueryHandler: dataset is empty or invalid");
    }

    std::cout << "T-BLAEQ: building index from in-memory PointCloud (N=" << dataset.size
              << " D=" << dataset.dim << " name=" << name << ")\n";
    const BuildPlan plan = adjustBuildPlan(static_cast<size_t>(dataset.size), height, ratios);
    if (plan.truncated) {
        std::cout << "Truncating build plan to avoid degenerate top layer.\n";
    }
    printBuildConfig(plan);

    const auto t0 = std::chrono::steady_clock::now();
    idx_.reset(IndexBuilder::build(dataset.data, static_cast<size_t>(dataset.size),
                                   static_cast<size_t>(dataset.dim), name, forceUseCPU, plan.height, plan.ratios));
    idx_->datasetName = name;
    const auto t1 = std::chrono::steady_clock::now();
    Chrono::printElapsed("Index build total", t0, t1);
}

QueryHandler::QueryHandler(size_t N, size_t D, double valMin, double valMax, bool isInt, size_t height,
                           const std::vector<size_t>& ratios, uint64_t seed, double sigmaDivisor,
                           const std::string& name) {
    std::cout << "T-BLAEQ: generating random index (N=" << N << " D=" << D << " range=["
              << valMin << "," << valMax << "] " << (isInt ? "int" : "float") << ")\n";
    const BuildPlan plan = adjustBuildPlan(N, height, ratios);
    if (plan.truncated) {
        std::cout << "Truncating build plan to avoid degenerate top layer.\n";
    }
    std::cout << "RandomKmeans config: height=" << plan.height << " ratios=[";
    for (size_t i = 0; i < plan.ratios.size(); ++i) {
        if (i) {
            std::cout << ",";
        }
        std::cout << plan.ratios[i];
    }
    std::cout << "] seed=" << seed << " sigmaDivisor=" << sigmaDivisor << "\n";

    const auto t0 = std::chrono::steady_clock::now();
    idx_.reset(IndexBuilder::buildRandom(N, D, valMin, valMax, isInt, name, plan.height, plan.ratios, seed, sigmaDivisor));
    idx_->datasetName = "Gen" + std::to_string(N) + "D" + std::to_string(D);
    const auto t1 = std::chrono::steady_clock::now();
    Chrono::printElapsed("Index build total", t0, t1);
}

QueryHandler::QueryHandler(const std::string& indexPath, bool loadFromIndex) {
    if (!loadFromIndex) {
        throw std::invalid_argument("Use QueryHandler(datasetPath) to build a new index.");
    }

    std::cout << "T-BLAEQ: loading index from " << indexPath << "\n";
    const auto t0 = std::chrono::steady_clock::now();
    idx_.reset(IndexSerializer::load(indexPath));
    idx_->datasetName = extractDatasetName(indexPath);
    const auto t1 = std::chrono::steady_clock::now();
    Chrono::printElapsed("Index load", t0, t1);
}

void QueryHandler::saveIndex(const std::string& dirPath) const {
    std::cout << "Saving index to: " << dirPath << "\n";
    const auto t0 = std::chrono::steady_clock::now();
    IndexSerializer::save(*idx_, dirPath);
    const auto t1 = std::chrono::steady_clock::now();
    Chrono::printElapsed("Index save", t0, t1);
}

void QueryHandler::doPrepare(const IndexPolicy& policy, const std::string& label) {
    std::cout << label << "\n";
    const auto t0 = std::chrono::steady_clock::now();

    idx_->activePolicy = policy;
    strategy_ = PolicyScheduler::make(*idx_, policy);
    strategy_->prepare(std::cout);

    const auto t1 = std::chrono::steady_clock::now();
    Chrono::printElapsed("Prepare total", t0, t1);
    prepared_ = true;
}

void QueryHandler::prepareForQuery() {
    if (prepared_) {
        return;
    }
    idx_->computeStats();
    const IndexPolicy policy = PolicyScheduler::recommend(*idx_);
    doPrepare(policy, "Preparing for query (auto policy selection)...");
}

void QueryHandler::prepareForQuery(IndexPolicy policy) {
    if (prepared_) {
        return;
    }
    doPrepare(policy, "Preparing for query with manual policy override...");
}

void QueryHandler::prepareForQuery(LevelPolicy level) {
    if (prepared_) {
        return;
    }
    const size_t n = idx_->intervals;
    prepareForQuery(IndexPolicy::uniform(n, level));
}

void QueryHandler::ensurePrepared() {
    if (!prepared_) {
        prepareForQuery();
    }
}

void QueryHandler::printMemoryStats(std::ostream& os) const {
    if (!prepared_) {
        os << "[MemoryStats] Not yet prepared. Call prepareForQuery() first.\n";
        return;
    }
    idx_->stats.print(os);
}

QueryResult QueryHandler::performQuery(const std::string& queryPath, QueryType qType, bool saveFineMesh,
                                       int maxQueryCount, size_t K) {
    ensurePrepared();

    const Query queryData =
        (qType == QueryType::POINT) ? loadQueryPointFromFile(queryPath) : loadQueryRangeFromFile(queryPath);

    if (queryData.type != qType) {
        std::cerr << "QueryHandler: query file type mismatch\n";
        QueryResult err;
        err.errorCode = 1;
        return err;
    }

    QueryEngine::RunConfig cfg;
    cfg.saveFineMesh = saveFineMesh;
    cfg.maxQueryCount = maxQueryCount;
    cfg.K = K;

    return QueryEngine::run(*idx_, *strategy_, queryData, cfg);
}

QueryResult QueryHandler::performSingleKNNQuery(const std::vector<double>& queryPoint, size_t k, bool saveFineMesh) {
    ensurePrepared();

    Query queryData(1, static_cast<int>(idx_->D), QueryType::POINT);
    queryData.setQueryPoint(0, queryPoint);

    QueryEngine::RunConfig cfg;
    cfg.saveFineMesh = saveFineMesh;
    cfg.maxQueryCount = 1;
    cfg.K = k;

    return QueryEngine::run(*idx_, *strategy_, queryData, cfg);
}

QueryResult QueryHandler::performSingleRangeQuery(const std::vector<double>& queryUpperBound,
                                                  const std::vector<double>& queryLowerBound,
                                                  bool saveFineMesh) {
    ensurePrepared();

    Query queryData(1, static_cast<int>(idx_->D), QueryType::RANGE);
    queryData.setQueryRange(0, queryLowerBound, queryUpperBound);
    queryData.queryRangeInfo = "single-range-query";

    QueryEngine::RunConfig cfg;
    cfg.saveFineMesh = saveFineMesh;
    cfg.maxQueryCount = 1;

    return QueryEngine::run(*idx_, *strategy_, queryData, cfg);
}
