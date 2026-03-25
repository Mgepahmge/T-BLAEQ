/**
 * @file IndexBuilder.cuh
 * @brief Constructs a hierarchical mesh index from a flat dataset.
 *
 * @details IndexBuilder runs KMeans at each level to build the coarsening
 * hierarchy, constructs the P-tensor (CSC-format prolongation operator) that
 * maps fine points to centroids, and computes the per-centroid max-radius
 * array used by the pruning kernels.  It has zero coupling to query logic
 * or serialisation.
 */

#pragma once

#include <string>
#include "IndexData.cuh"
#include "src/Data_Structures/Data_Structures.cuh"

/**
 * @class IndexBuilder
 * @brief Builds an IndexData from a flat host-memory dataset.
 *
 * @details The builder iterates from the finest level to the coarsest,
 * running KMeans at each step to produce centroids and building the
 * corresponding P-tensor and max-radius array.  The resulting IndexData
 * contains only host-side data; no device uploads are performed here.
 * All device management is deferred to the strategy's prepare() call.
 */
class IndexBuilder {
public:
    static constexpr size_t kDefaultHeight = 4; //!< Default number of mesh levels.
    static const std::vector<size_t> kDefaultRatios; //!< Default coarsening ratios {100, 50, 20}.

    /*!
     * @brief Build the complete hierarchical index from a flat dataset.
     *
     * @details Runs KMeans (height - 1) times, building one P-tensor and one
     * max-radius array per level.  The coarsest mesh is stored in
     * IndexData::coarsestMesh.  The returned object is heap-allocated and
     * caller-owned.
     *
     * @param[in] data    Host pointer to the dataset in AOS layout (N * D doubles).
     * @param[in] N       Total number of data points.
     * @param[in] D       Dimensionality of each point.
     * @param[in] name    Human-readable dataset label stored in IndexData for logging.
     * @param[in] forceUseCPU
     * @param[in] height  Total number of mesh levels including the finest.
     * @param[in] ratios  Coarsening ratio per level; length must equal height - 1.
     * @return Heap-allocated IndexData containing the fully built host-side index.
     */
    static IndexData* build(const double* data, size_t N, size_t D, const std::string& name, bool forceUseCPU = false,
                            size_t height = kDefaultHeight, const std::vector<size_t>& ratios = kDefaultRatios);

private:
    /*!
     * @brief Compute the number of centroids for a given data count and ratio.
     *
     * @param[in] dataNums Number of data points at the current level.
     * @param[in] ratio    Coarsening ratio for this level.
     * @return Number of centroids to produce.
     */
    static size_t computeCentroidCount(size_t dataNums, size_t ratio);
};
