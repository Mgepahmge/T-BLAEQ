#include "Setup.cuh"
#include "src/func.hpp"

size_t Compute_Layer_nums(size_t /*N*/) { return 4; }

size_t Compute_Centroid_nums(size_t dataNums, size_t ratio) {
    if (ratio == 0 || dataNums / ratio == 0) {
        return 1;
    }
    return dataNums / ratio;
}

SparseTensorCscFormat* Genenate_One_P_Tensor(size_t D, size_t pRowLen, size_t pColLen, CUDAKmeans* kmeans,
                                             size_t*& map) {
    assert(pRowLen > 0);
    assert(pColLen > 0);

    // Retrieve KMeans results
    const auto& coarseMesh = kmeans->getCentroids();
    const auto& fineMesh = kmeans->getdatas();
    const auto& labels = kmeans->getLabels();

    // Count NNZ per column (number of fine-mesh points per centroid)
    std::vector<size_t> nnzPerCol(pColLen, 0);
    std::for_each(labels.begin(), labels.end(), [&nnzPerCol](int id) { ++nnzPerCol[id]; });

    const size_t maxNnzPerCol = *std::max_element(nnzPerCol.begin(), nnzPerCol.end());
    assert(maxNnzPerCol > 0);
    std::cout << "max_nnz_per_col: " << maxNnzPerCol << "\n";

    // Temporary batch buffer (+10 to avoid boundary issues)
    auto* batchBuf = new double[D * (maxNnzPerCol + 10)]();

    // Allocate the CSC tensor
    auto* pTensor = new SparseTensorCscFormat(D, pRowLen, pColLen, nnzPerCol);

    // Build the sort-to-original map
    std::vector<size_t> sortedIdx = Sort::Sorted_Layer_With_Original_idxs(labels);
    map = new size_t[sortedIdx.size()];
    std::copy(sortedIdx.begin(), sortedIdx.end(), map);

    // Fill value batches: for each centroid column, write the ratio
    // (fine[j] / coarse[i]) component-wise into the CSC value array.
    const size_t* colRes = pTensor->get_col_res();
    for (size_t col = 0; col < pColLen; ++col) {
        const size_t beg = colRes[col];
        const size_t end = colRes[col + 1];
        const auto& cVal = coarseMesh[col];
        size_t wIdx = 0;

        for (size_t pos = beg; pos < end; ++pos) {
            const size_t origId = sortedIdx[pos];
            const auto& fVal = fineMesh[origId];
            for (size_t d = 0; d < D; ++d) {
                if (Comp::isZero(cVal[d])) {
                    std::cout << "ERROR: division by zero in P-tensor build\n";
                    assert(false);
                }
                batchBuf[wIdx * D + d] = fVal[d] / cVal[d];
            }
            ++wIdx;
        }
        pTensor->Insert_One_Batch(batchBuf, beg, end);
    }

    delete[] batchBuf;
    return pTensor;
}

double* Compute_Max_Radius(size_t D, const size_t* centroidColRes, const size_t* sortToOriginalMap,
                           CUDAKmeans* kmeans) {
    const auto& coarseMesh = kmeans->getCentroids();
    const auto& fineMesh = kmeans->getdatas();
    const size_t nCentroids = coarseMesh.size();

    auto* radius = new double[nCentroids];

    for (size_t i = 0; i < nCentroids; ++i) {
        const size_t beg = centroidColRes[i];
        const size_t end = centroidColRes[i + 1];
        const auto& centroid = coarseMesh[i];
        double maxDist = 0.0;

        for (size_t pos = beg; pos < end; ++pos) {
            const size_t absIdx = sortToOriginalMap[pos];
            maxDist = std::max(maxDist, dist::euclidean(centroid, fineMesh[absIdx]));
        }
        radius[i] = maxDist;
    }

    return radius;
}

std::string getQueryTypeString(QueryType qType) {
    switch (qType) {
    case QueryType::RANGE:
        return "RANGE";
    case QueryType::POINT:
        return "POINT";
    default:
        return "UNKNOWN";
    }
}
