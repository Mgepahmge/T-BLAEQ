#include <iostream>
#include <stdexcept>
#include "QueryHandler.cuh"
#include "src/Data_Structures/File.cuh"
#include "src/func.hpp"
#include "src/utils/Utils.cuh"

QueryHandler::QueryHandler(const bool forceUseCPU, const std::string& datasetPath,
                           const size_t height, const std::vector<size_t>& ratios) {
    std::cout << "T-BLAEQ: building index from " << datasetPath << "\n";
    std::cout << "Build config: height=" << height << " ratios=[";
    for (size_t i = 0; i < ratios.size(); ++i) {
        if (i) {
            std::cout << ",";
        }
        std::cout << ratios[i];
    }
    std::cout << "]\n";

    const std::string name = extractDatasetName(datasetPath);
    const PointCloud dataset = loadFromFile(datasetPath);

    const auto t0 = std::chrono::steady_clock::now();
    idx_.reset(IndexBuilder::build(dataset.data, static_cast<size_t>(dataset.size), static_cast<size_t>(dataset.dim),
                                   name, forceUseCPU, height, ratios));
    const auto t1 = std::chrono::steady_clock::now();
    Chrono::printElapsed("Index build total", t0, t1);
}

QueryHandler::QueryHandler(size_t N, size_t D, double valMin, double valMax, bool isInt, size_t height,
                           const std::vector<size_t>& ratios, uint64_t seed, double sigmaDivisor,
                           const std::string& name) {
    std::cout << "T-BLAEQ: generating random index (N=" << N << " D=" << D << " range=["
              << valMin << "," << valMax << "] " << (isInt ? "int" : "float") << ")\n";
    std::cout << "RandomKmeans config: height=" << height << " ratios=[";
    for (size_t i = 0; i < ratios.size(); ++i) {
        if (i) {
            std::cout << ",";
        }
        std::cout << ratios[i];
    }
    std::cout << "] seed=" << seed << " sigmaDivisor=" << sigmaDivisor << "\n";

    const auto t0 = std::chrono::steady_clock::now();
    idx_.reset(IndexBuilder::buildRandom(N, D, valMin, valMax, isInt, name, height, ratios, seed, sigmaDivisor));
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
