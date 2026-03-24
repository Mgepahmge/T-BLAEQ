#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include "IndexBuilder.cuh"
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

IndexData* IndexBuilder::build(const double* data, size_t N, size_t D, const std::string& name, size_t height,
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
    idx->dMaps.resize(height); // DeviceBuffer is default-constructible (empty)

    // Iterative KMeans + P-tensor construction
    //   level 0 = coarsest,  level (height-1) = finest
    //   We build from fine -> coarse, so the first KMeans call operates on
    //   the full dataset (finest mesh).
    constexpr size_t kMaxIter = 7;

    auto* kmeans = new CUDAKmeans(data, N, D, /*isAos=*/true);

    std::cout << "Building hierarchical index (" << height << " levels, " << idx->intervals << " P-tensors)\n"
              << "Level 0 = coarsest, level " << (height - 1) << " = finest\n\n";

    for (size_t i = 0; i < idx->intervals; ++i) {
        const size_t dataNums = kmeans->getdatas().size();
        const size_t centroidNums = computeCentroidCount(dataNums, ratios[i]);
        const size_t fineIdx = idx->intervals - i; // current fine level
        const size_t coarseIdx = fineIdx - 1; // target coarse level
        const size_t pTensorIdx = coarseIdx; // slot in pTensors[]

        idx->meshSizes.push_back(dataNums);

        printf("Mesh_%zu -> Mesh_%zu  KMeans  (%zu -> %zu points)\n", fineIdx, coarseIdx, dataNums, centroidNums);

        auto t0 = std::chrono::steady_clock::now();
        kmeans->run(centroidNums, kMaxIter);
        auto t1 = std::chrono::steady_clock::now();
        Chrono::printElapsed("  KMeans", t0, t1);

        // Build one P-tensor (fine -> coarse prolongation operator in CSC).
        // Also produces the sort-to-original map for the fine level.
        auto tP0 = std::chrono::steady_clock::now();
        idx->pTensors[pTensorIdx] = Genenate_One_P_Tensor(D, dataNums, centroidNums, kmeans, idx->maps[fineIdx]);
        assert(idx->maps[fineIdx] != nullptr);
        auto tP1 = std::chrono::steady_clock::now();
        Chrono::printElapsed("  P-tensor build", tP0, tP1);

        // Compute max radius for the coarse level.
        auto tR0 = std::chrono::steady_clock::now();
        idx->meshMaxRadius[pTensorIdx] =
            Compute_Max_Radius(D, idx->pTensors[pTensorIdx]->get_col_res(), idx->maps[fineIdx], kmeans);
        auto tR1 = std::chrono::steady_clock::now();
        Chrono::printElapsed("  MaxRadius", tR0, tR1);

        // On last iteration: record coarsest mesh + centroid count.
        if (i + 2 == height) {
            const auto& centroids = kmeans->getCentroids();
            idx->meshSizes.push_back(centroids.size());
            idx->coarsestMesh = new GridAsSparseMatrix(centroids, 0, centroids.size());
        }

        kmeans->reset();
        std::cout << "\n";
    }

    delete kmeans;

    // Print mesh size summary
    std::cout << "Mesh sizes (finest -> coarsest): " << idx->meshSizes[0];
    for (size_t i = 1; i < idx->meshSizes.size(); ++i) {
        std::cout << " -> " << idx->meshSizes[i];
    }
    std::cout << "\n";

    // Device upload is deferred to IndexData::prepareForQuery(),
    // which selects the appropriate per-level policy first.

    return idx;
}
