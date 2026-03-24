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
#include "src/Kmeans/CUDAKmeans.cuh"
#include "src/Query/check.cuh"

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
 * @param[in]  kmeans  Pointer to a CUDAKmeans object after run() has been called.
 * @param[out] map     Newly allocated array of length pRowLen mapping sorted
 *                     positions back to original point indices. Caller owns.
 * @return Newly allocated CSC tensor. Caller owns.
 */
SparseTensorCscFormat* Genenate_One_P_Tensor(size_t D, size_t pRowLen, size_t pColLen, CUDAKmeans* kmeans,
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
 * @param[in] kmeans            Pointer to the same CUDAKmeans object after run().
 * @return Newly allocated array of length nCentroids. Caller owns.
 */
double* Compute_Max_Radius(size_t D, const size_t* centroidColRes, const size_t* sortToOriginalMap, CUDAKmeans* kmeans);

/*!
 * @brief Convert a QueryType enum value to a human-readable string.
 *
 * @param[in] qType The query type to convert.
 * @return "RANGE" or "POINT".
 */
std::string getQueryTypeString(QueryType qType);
