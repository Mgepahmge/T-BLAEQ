#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "QueryEngine.cuh"
#include "src/func.hpp"
#include "src/utils/NVTXProfiler.cuh"
#include "src/utils/Utils.cuh"
#include "strategies/L0Strategy.cuh"
#include "strategies/L1Strategy.cuh"
#include "strategies/L2Strategy.cuh"
#include "strategies/L3Strategy.cuh"
#include "strategies/StrategyCommon.cuh"

QueryResult::~QueryResult() {
    for (size_t i = 0; i < fineMesh.size(); ++i) {
        const bool ownsIds = (i < fineMeshOwnsIds.size()) ? fineMeshOwnsIds[i] : true;
        const bool onHost = (i < fineMeshOnHost.size()) ? fineMeshOnHost[i] : false;
        if (onHost) {
            delete[] fineMesh[i]->get_ids_();
            delete[] fineMesh[i]->get_vals_();
            delete fineMesh[i];
        }
        else {
            const bool ownsVals = true; // fineMesh always owns vals (L0 grids not saved)
            freeDeviceSparseGrid(fineMesh[i], ownsIds, ownsVals);
        }
    }
}

static double calcRangeLogVolume(const double* lo, const double* hi, size_t dim) {
    double v = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double s = hi[i] - lo[i];
        if (s <= 0.0) {
            return -std::numeric_limits<double>::infinity();
        }
        v += std::log(s);
    }
    return v;
}

static double calcKnnLogVolume(const double*, const SparseGrid*) { return 1.0; }

static std::string formatLogVolume(double lv) {
    if (std::isinf(lv)) {
        return (lv < 0) ? "0" : "inf";
    }
    if (std::isnan(lv)) {
        return "nan";
    }
    const double e = std::floor(lv / std::log(10.0));
    const double m = std::pow(10.0, lv / std::log(10.0) - e);
    std::ostringstream o;
    o << std::fixed << std::setprecision(2) << m << 'E' << (e >= 0 ? "+" : "") << static_cast<long long>(e);
    return o.str();
}

LevelResult QueryEngine::runSingleQuery(IndexData& idx, QueryType qType, const double* lo, const double* hi,
                                        const double* queryPoint, size_t K) {
    // L3 does not set permanentDataOnDevice; each strategy enforces its own
    // pre-conditions in runLevel().

    SparseGrid coarsestView{idx.coarsestMesh->get_num_rows(), idx.coarsestMesh->get_dimensions(),
                            idx.coarsestMesh->get_nnz_nums(), idx.dCoarsestMeshIds.data(),
                            idx.dCoarsestMeshVals.data()};

    SparseGrid* currentGrid = &coarsestView;
    bool currentOwned = false;
    LevelResult currentResult{};

    for (size_t l = 0; l < idx.intervals; ++l) {
        auto levelStrategy = PolicyScheduler::makeForLevel(idx, l);
        LevelResult result = levelStrategy->runLevel(l, currentGrid, qType, lo, hi, queryPoint, K);

        // Free the previous level's grid.
        // Each strategy is responsible for handling its own input correctly
        // (e.g. L3 downloads device grids internally before processing).
        // QueryEngine only needs to free the grid it previously received.
        if (currentOwned) {
            if (currentResult.onHost) {
                // Host-resident grid (L3 output): ids_/vals_ are new[]-allocated.
                // ownsIds == true always for L3 outputs.
                delete[] currentGrid->get_ids_();
                delete[] currentGrid->get_vals_();
                delete currentGrid;
            }
            else {
                freeDeviceSparseGrid(currentGrid, currentResult.ownsIds, currentResult.ownsVals);
            }
        }

        currentGrid = result.grid;
        currentOwned = true;
        currentResult = result;
    }

    return currentResult;
}

QueryResult QueryEngine::run(IndexData& idx, IQueryStrategy& strategy, const Query& queryData, const RunConfig& cfg) {
    (void)strategy; // strategy was used for prepare(); per-level dispatch via makeForLevel()

    QueryResult result;
    result.type = queryData.type;
    result.datasetName = idx.datasetName;
    result.datasetSize = idx.N;
    result.datasetDim = idx.D;

    const bool isRange = (queryData.type == QueryType::RANGE);
    const bool isKnn = (queryData.type == QueryType::POINT);
    int count = std::min(cfg.maxQueryCount, queryData.length);

    if (cfg.saveFineMesh) {
        result.fineMesh.reserve(count);
    }
    result.queryRangeVolume.reserve(count);
    result.fineMeshSize.reserve(count);

    std::vector<double> lo(idx.D), hi(idx.D), qpt(idx.D);
    long totalUs = 0;

    if (isKnn) {
        std::cout << "KNN K=" << cfg.K << "\n";
    }

    long maxUs = std::numeric_limits<long>::min();
    long minUs = std::numeric_limits<long>::max();
    for (int i = 0; i < count; ++i) {
        std::cout << "Query " << i << "\n";

        if (isRange) {
            auto [lower, upper] = queryData.getQueryRange(i);
            std::copy(lower.begin(), lower.end(), lo.data());
            std::copy(upper.begin(), upper.end(), hi.data());
        }
        else {
            auto point = queryData.getQueryPoint(i);
            std::copy(point.begin(), point.end(), qpt.data());
        }

        const std::string pname = (isKnn ? std::string("KNN") : std::string("Range")) + "Query_N" +
            std::to_string(idx.N) + "_D" + std::to_string(idx.D);
        NvtxProfiler profiler(pname.c_str(), NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Red);

        const auto t0 = std::chrono::steady_clock::now();

        LevelResult finalResult = runSingleQuery(idx, queryData.type, lo.data(), hi.data(), qpt.data(), cfg.K);

        profiler.release();
        const auto t1 = std::chrono::steady_clock::now();

        const long us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        if (us > maxUs) {
            maxUs = us;
        }
        if (us < minUs) {
            minUs = us;
        }
        totalUs += us;
        std::cout << "  -> " << (us / 1000.0) << " ms\n";

        result.fineMeshSize.push_back(finalResult.grid->get_nnz_nums());

        if (isRange) {
            result.queryRangeVolume.push_back(calcRangeLogVolume(lo.data(), hi.data(), idx.D));
        }
        else {
            result.queryRangeVolume.push_back(calcKnnLogVolume(qpt.data(), finalResult.grid));
        }

        if (cfg.saveFineMesh) {
            result.fineMesh.push_back(finalResult.grid);
            result.fineMeshOwnsIds.push_back(finalResult.ownsIds);
            result.fineMeshOnHost.push_back(finalResult.onHost);
        }
        else {
            if (finalResult.onHost) {
                delete[] finalResult.grid->get_ids_();
                delete[] finalResult.grid->get_vals_();
                delete finalResult.grid;
            }
            else {
                freeDeviceSparseGrid(finalResult.grid, finalResult.ownsIds, finalResult.ownsVals);
            }
        }
    }

    if (count > 2) {
        totalUs = totalUs - maxUs;
        totalUs = totalUs - minUs;
        count = count - 2;
    }

    result.totalTimeUs = totalUs;
    result.queryCount = static_cast<size_t>(count);

    const double avgMs = count > 0 ? static_cast<double>(totalUs) / count / 1000.0 : 0.0;
    std::cout << "Average: " << avgMs << " ms\n"
              << "Total:   " << (totalUs / 1000.0) << " ms\n";
    return result;
}

void saveQueryResult(const QueryResult& result, const std::string& outputFile) {
    if (result.errorCode != 0) {
        std::cerr << "QueryResult errorCode=" << result.errorCode << " -- skipping.\n";
        return;
    }

    const bool exists = std::filesystem::exists(outputFile);
    std::ofstream out(outputFile, exists ? std::ios::app : std::ios::out);
    if (!out) {
        std::cerr << "Cannot open: " << outputFile << "\n";
        return;
    }

    if (!exists) {
        out << "Dataset,Size,Dim,Query Type,Query Parameter,"
               "Query Count,Total Time (ms),Avg Time (ms),"
               "Median Log Volume,Avg Range Volume,Avg Fine Mesh\n";
    }

    const double totalMs = result.totalTimeUs / 1000.0;
    const double avgMs = result.queryCount > 0 ? totalMs / static_cast<double>(result.queryCount) : 0.0;

    std::string medStr = "N/A", fmtStr = "N/A";
    if (!result.queryRangeVolume.empty()) {
        std::vector<double> sv = result.queryRangeVolume;
        std::sort(sv.begin(), sv.end());
        const size_t n = sv.size();
        const double med = (n % 2 == 0) ? (sv[n / 2 - 1] + sv[n / 2]) / 2.0 : sv[n / 2];
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3) << med;
        medStr = oss.str();
        fmtStr = formatLogVolume(med);
    }

    std::string avgFineStr = "N/A";
    if (!result.fineMeshSize.empty()) {
        double sum = 0.0;
        for (auto s : result.fineMeshSize) {
            sum += static_cast<double>(s);
        }
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << sum / static_cast<double>(result.fineMeshSize.size());
        avgFineStr = oss.str();
    }

    out << result.datasetName << ',' << result.datasetSize << ',' << result.datasetDim << ','
        << ((result.type == QueryType::POINT) ? "POINT (KNN)" : "RANGE") << ',' << result.queryParam << ','
        << result.queryCount << ',' << std::fixed << std::setprecision(3) << totalMs << ',' << std::fixed
        << std::setprecision(6) << avgMs << ',' << medStr << ',' << fmtStr << ',' << avgFineStr << '\n';

    std::cout << "Result saved to: " << outputFile << "\n";
}
