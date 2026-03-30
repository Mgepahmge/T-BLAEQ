#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include "IndexBuilder.cuh"
#include "src/Kmeans/CPUKmeans.cuh"
#include "src/Kmeans/RandomKmeans.cuh"
#include "src/Kmeans/CUDAKmeans.cuh"
#include "src/Setup/Setup.cuh"
#include "src/func.hpp"

const std::vector<size_t> IndexBuilder::kDefaultRatios = {100, 50, 20};

size_t IndexBuilder::computeCentroidCount(size_t dataNums, size_t ratio) {
    if (ratio == 0 || dataNums / ratio == 0) {
        return 1;
    }
    return dataNums / ratio;
}

/*!
 * @brief Return true when the dataset fits in available device memory.
 *
 * @details Uses a conservative 85% safety factor (same as the query
 * scheduler) to avoid OOM during KMeans initialisation and iteration.
 * The estimate accounts for the flat dataset array plus approximately
 * one additional copy used by cuVS internally during fitting.
 *
 * @param[in] N    Number of data points.
 * @param[in] D    Data dimensionality.
 * @return True when GPU KMeans can be used safely.
 */
static bool datasetFitsInGPU(size_t N, size_t D) {
    size_t freeMem = 0;
    size_t totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);

    // Dataset itself + ~1x overhead for cuVS internal buffers
    const size_t required = static_cast<size_t>(N * D * sizeof(double) * 2.0);
    constexpr double kSafetyFactor = 0.85;
    return required <= static_cast<size_t>(static_cast<double>(freeMem) * kSafetyFactor);
}

IndexData* IndexBuilder::build(const double* data, size_t N, size_t D, const std::string& name, const bool forceUseCPU, size_t height,
                               const std::vector<size_t>& ratios) {
    assert(height >= 2);
    assert(ratios.size() == height - 1);

    auto* idx = new IndexData();
    idx->D = D;
    idx->N = N;
    idx->height = height;
    idx->intervals = height - 1;
    idx->isAosArch = true;
    idx->datasetName = name;
    idx->ratios = ratios;

    idx->pTensors.resize(idx->intervals, nullptr);
    idx->meshMaxRadius.resize(idx->intervals, nullptr);
    idx->maps.resize(height, nullptr);
    idx->dMaps.resize(height);

    constexpr size_t kMaxIter = 7;

    // Select KMeans backend based on available device memory.
    // CPUKmeans and CUDAKmeans expose an identical interface so the build
    // loop below is backend-agnostic.
    bool useGPU = datasetFitsInGPU(N, D);

    if (forceUseCPU) {
        useGPU = false;
    }

    if (useGPU) {
        std::cout << "KMeans backend: GPU (cuVS)\n";
    } else {
        std::cout << "KMeans backend: CPU fallback (dataset too large for device memory)\n";
    }

    std::cout << "Building hierarchical index (" << height << " levels, " << idx->intervals << " P-tensors)\n"
              << "Level 0 = coarsest, level " << (height - 1) << " = finest\n\n";

    // Use a type-erased lambda to run the build loop with either backend,
    // avoiding code duplication.
    auto runBuild = [&](auto& kmeans) {
        for (size_t i = 0; i < idx->intervals; ++i) {
            const size_t dataNums = kmeans.getdatas().size();
            const size_t centroidNums = computeCentroidCount(dataNums, ratios[i]);
            const size_t fineIdx = idx->intervals - i;
            const size_t coarseIdx = fineIdx - 1;
            const size_t pTensorIdx = coarseIdx;

            idx->meshSizes.push_back(dataNums);

            printf("Mesh_%zu -> Mesh_%zu  KMeans  (%zu -> %zu points)\n", fineIdx, coarseIdx, dataNums,
                   centroidNums);

            auto t0 = std::chrono::steady_clock::now();
            kmeans.run(centroidNums, kMaxIter);
            auto t1 = std::chrono::steady_clock::now();
            Chrono::printElapsed("  KMeans", t0, t1);

            auto tP0 = std::chrono::steady_clock::now();
            idx->pTensors[pTensorIdx] =
                Genenate_One_P_Tensor(D, dataNums, centroidNums, &kmeans, idx->maps[fineIdx]);
            assert(idx->maps[fineIdx] != nullptr);
            auto tP1 = std::chrono::steady_clock::now();
            Chrono::printElapsed("  P-tensor build", tP0, tP1);

            auto tR0 = std::chrono::steady_clock::now();
            idx->meshMaxRadius[pTensorIdx] =
                Compute_Max_Radius(D, idx->pTensors[pTensorIdx]->get_col_res(), idx->maps[fineIdx], &kmeans);
            auto tR1 = std::chrono::steady_clock::now();
            Chrono::printElapsed("  MaxRadius", tR0, tR1);

            if (i + 2 == height) {
                const auto& centroids = kmeans.getCentroids();
                idx->meshSizes.push_back(centroids.size());
                idx->coarsestMesh = new GridAsSparseMatrix(centroids, 0, centroids.size());
            }

            kmeans.reset();
            std::cout << "\n";
        }
    };

    if (useGPU) {
        try
        {
            CUDAKmeans kmeans(data, N, D, /*isAos=*/true);
            runBuild(kmeans);
        } catch (std::exception& e) {
            std::cout << "Error building kmeans: " << e.what() << "\n";
            std::cout << "Trying CPU Kmeans..." << std::endl;
            CPUKmeans kmeans(data, N, D, true);
            runBuild(kmeans);
        }
    } else {
        CPUKmeans kmeans(data, N, D, /*isAos=*/true);
        runBuild(kmeans);
    }

    // Print mesh size summary
    std::cout << "Mesh sizes (finest -> coarsest): " << idx->meshSizes[0];
    for (size_t i = 1; i < idx->meshSizes.size(); ++i) {
        std::cout << " -> " << idx->meshSizes[i];
    }
    std::cout << "\n";

    return idx;
}

IndexData* IndexBuilder::buildRandom(size_t N, size_t D, double valMin, double valMax, bool isInt,
                                     const std::string& name, size_t height,
                                     const std::vector<size_t>& ratios,
                                     const uint64_t seed, const double sigmaDivisor) {
    assert(height >= 2);
    assert(ratios.size() == height - 1);

    auto* idx = new IndexData();
    idx->D = D;
    idx->N = N;
    idx->height = height;
    idx->intervals = height - 1;
    idx->isAosArch = true;
    idx->datasetName = name;
    idx->ratios = ratios;

    idx->pTensors.resize(idx->intervals, nullptr);
    idx->meshMaxRadius.resize(idx->intervals, nullptr);
    idx->maps.resize(height, nullptr);
    idx->dMaps.resize(height);

    std::cout << "KMeans backend: Random (synthetic data generation)\n";
    std::cout << "Building hierarchical index (" << height << " levels, " << idx->intervals << " P-tensors)\n"
              << "Level 0 = coarsest, level " << (height - 1) << " = finest\n\n";

    RandomKmeans::Config randomCfg;
    randomCfg.seed = seed;
    randomCfg.sigmaDivisor = sigmaDivisor;
    RandomKmeans kmeans(N, D, ratios, valMin, valMax, isInt, randomCfg);

    auto runBuild = [&](auto& km) {
        for (size_t i = 0; i < idx->intervals; ++i) {
            const size_t dataNums = km.getdatas().size();
            const size_t centroidNums = computeCentroidCount(dataNums, ratios[i]);
            const size_t fineIdx = idx->intervals - i;
            const size_t coarseIdx = fineIdx - 1;
            const size_t pTensorIdx = coarseIdx;

            idx->meshSizes.push_back(dataNums);

            printf("Mesh_%zu -> Mesh_%zu  Random  (%zu -> %zu points)\n", fineIdx, coarseIdx, dataNums,
                   centroidNums);

            auto t0 = std::chrono::steady_clock::now();
            km.run(centroidNums, 0);
            auto t1 = std::chrono::steady_clock::now();
            Chrono::printElapsed("  Random gen", t0, t1);

            auto tP0 = std::chrono::steady_clock::now();
            idx->pTensors[pTensorIdx] =
                Genenate_One_P_Tensor(D, dataNums, centroidNums, &km, idx->maps[fineIdx]);
            assert(idx->maps[fineIdx] != nullptr);
            auto tP1 = std::chrono::steady_clock::now();
            Chrono::printElapsed("  P-tensor build", tP0, tP1);

            auto tR0 = std::chrono::steady_clock::now();
            idx->meshMaxRadius[pTensorIdx] =
                Compute_Max_Radius(D, idx->pTensors[pTensorIdx]->get_col_res(), idx->maps[fineIdx], &km);
            auto tR1 = std::chrono::steady_clock::now();
            Chrono::printElapsed("  MaxRadius", tR0, tR1);

            if (i + 2 == height) {
                const auto& centroids = km.getCentroids();
                idx->meshSizes.push_back(centroids.size());
                idx->coarsestMesh = new GridAsSparseMatrix(centroids, 0, centroids.size());
            }

            km.reset();
            std::cout << "\n";
        }
    };

    runBuild(kmeans);

    std::cout << "Mesh sizes (finest -> coarsest): " << idx->meshSizes[0];
    for (size_t i = 1; i < idx->meshSizes.size(); ++i) {
        std::cout << " -> " << idx->meshSizes[i];
    }
    std::cout << "\n";

    return idx;
}
