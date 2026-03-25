/**
 * @file Setup.cuh
 * @brief Index construction helpers used by IndexBuilder.
 *
 * @details Provides the functions that build one level of the hierarchical
 * index: computing centroid counts, generating the P-tensor in CSC format,
 * computing per-cluster max-radius values, and miscellaneous utilities.
 */

#pragma once

#include "src/Data_Structures/File.cuh"
#include "src/Kmeans/CPUKmeans.cuh"
#include "src/Kmeans/CUDAKmeans.cuh"
#include "src/Query/check.cuh"
#include "src/func.hpp"

/*!
 * @brief Return the fixed number of hierarchy levels.
 *
 * @details Currently hard-coded to 4: coarsest mesh at level 0,
 * finest mesh at level 3.
 *
 * @param[in] N Total number of data points (unused; reserved for future scaling).
 * @return Number of hierarchy levels.
 */
size_t Compute_Layer_nums(size_t N);

/*!
 * @brief Compute the number of centroids for one KMeans step.
 *
 * @param[in] dataNums Number of points at the current level.
 * @param[in] ratio    Downsampling ratio (e.g. 100 means dataNums / 100 centroids).
 * @return Number of centroids, guaranteed to be at least 1.
 */
size_t Compute_Centroid_nums(size_t dataNums, size_t ratio);

/*!
 * @brief Build one P-tensor in CSC format and produce the sort-to-original map.
 *
 * @details Constructs the prolongation operator P for one hierarchy level.
 * P has pRowLen rows (fine-mesh points) and pColLen columns (centroids).
 * Each fine point belongs to exactly one centroid, so P has exactly pRowLen
 * non-zeros.  The sort-to-original map records how fine points were reordered
 * by KMeans cluster assignment.
 *
 * @param[in]  D       Data dimensionality.
 * @param[in]  pRowLen Number of rows in P (fine-mesh NNZ count).
 * @param[in]  pColLen Number of columns in P (centroid count).
 * @param[in]  kmeans  Pointer to a KMeans object (CUDAKmeans or CPUKmeans) after run() has been called.
 * @param[out] map     Newly allocated array of length pRowLen mapping sorted
 *                     positions back to original point indices. Caller owns.
 * @return Newly allocated CSC tensor. Caller owns.
 */
template <typename KMeans>
SparseTensorCscFormat* Genenate_One_P_Tensor(size_t D, size_t pRowLen, size_t pColLen, KMeans* kmeans,
                                             size_t*& map);

/*!
 * @brief Compute the maximum Euclidean radius of each cluster.
 *
 * @details For each centroid c_i, finds the farthest assigned fine point and
 * stores that distance as the cluster max-radius.  The result is stored in
 * IndexData::meshMaxRadius[l] and uploaded to the device during pruning.
 *
 * @param[in] D                 Data dimensionality.
 * @param[in] centroidColRes    Column-offset array of the P-tensor CSC format
 *                              (length nCentroids + 1).
 * @param[in] sortToOriginalMap Sorted-to-original index map from Genenate_One_P_Tensor.
 * @param[in] kmeans            Pointer to the same KMeans object (CUDAKmeans or CPUKmeans) after run().
 * @return Newly allocated array of length nCentroids. Caller owns.
 */
template <typename KMeans>
double* Compute_Max_Radius(size_t D, const size_t* centroidColRes, const size_t* sortToOriginalMap,
                           KMeans* kmeans);

/*!
 * @brief Convert a QueryType enum value to a human-readable string.
 *
 * @param[in] qType The query type to convert.
 * @return "RANGE" or "POINT".
 */
std::string getQueryTypeString(QueryType qType);

// Template implementations (must be in header for instantiation at call sites)

template <typename KMeans>
SparseTensorCscFormat* Genenate_One_P_Tensor(size_t D, size_t pRowLen, size_t pColLen, KMeans* kmeans,
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

template <typename KMeans>
double* Compute_Max_Radius(size_t D, const size_t* centroidColRes, const size_t* sortToOriginalMap,
                           KMeans* kmeans) {
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
