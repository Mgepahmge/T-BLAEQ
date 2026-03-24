/**
 * @file CUDAKmeans.cuh
 * @brief GPU-accelerated KMeans clustering using the RAFT/cuVS library.
 *
 * @details CUDAKmeans wraps the cuVS KMeans implementation and manages data
 * upload to the device, cluster assignment, centroid download, and reset
 * between hierarchy levels.  It is used exclusively by IndexBuilder to
 * generate the coarsening hierarchy during index construction.
 */

#ifndef CUDA_KMEANS_H
#define CUDA_KMEANS_H

#include <cuvs/cluster/kmeans.hpp>
#include <memory>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <vector>

/**
 * @class CUDAKmeans
 * @brief Wrapper around the cuVS KMeans algorithm for hierarchical index construction.
 *
 * @details On construction the dataset is uploaded to the GPU.  Calling run()
 * performs KMeans clustering into k clusters.  The resulting labels and
 * centroids can then be read via the accessors and used to build the P-tensor
 * for the current hierarchy level.  reset() discards the current cluster
 * assignment so that run() can be called again for the next level.
 */
class CUDAKmeans {
public:
    using Point = std::vector<double>; //!< D-dimensional point as a double vector.

    /*!
     * @brief Upload the dataset to the GPU and prepare for clustering.
     *
     * @param[in] data   Host pointer to the dataset in AOS layout (N * dim doubles).
     * @param[in] N      Number of data points.
     * @param[in] dim    Data dimensionality.
     * @param[in] is_aos True when data is in AOS layout (default).
     */
    CUDAKmeans(const double* data, size_t N, size_t dim, bool is_aos = true);

    ~CUDAKmeans();

    /*!
     * @brief Run KMeans clustering into k clusters.
     *
     * @param[in] k         Number of clusters (centroids) to produce.
     * @param[in] max_iters Maximum number of KMeans iterations.
     */
    void run(size_t k, size_t max_iters = 100);

    void displayGroup();

    /*!
     * @brief Discard the current cluster assignment and free GPU cluster buffers.
     *
     * @details After reset(), run() can be called again with a different k.
     * The dataset remains on the GPU.
     */
    void reset();

    /*!
     * @brief Replace the dataset with new data and reset cluster state.
     *
     * @param[in] flat_data Host pointer to the new dataset (N * dim doubles, AOS).
     * @param[in] N         Number of points in the new dataset.
     * @param[in] dim       Dimensionality of the new dataset.
     */
    void clean_and_reset(const double* flat_data, size_t N, size_t dim);

    [[nodiscard]] const std::vector<size_t>& getLabels() const {
        return labels_;
    } /*!< Per-point cluster assignment (index into centroids). */
    [[nodiscard]] const std::vector<Point>& getCentroids() const {
        return centroids_;
    } /*!< Cluster centroid coordinates after run(). */
    [[nodiscard]] const std::vector<Point>& getdatas() const { return datas_; } /*!< Current level dataset points. */
    [[nodiscard]] size_t get_curr_layer_length() const {
        return datas_.size();
    } /*!< Number of points at the current level. */
    [[nodiscard]] size_t get_next_layer_length() const {
        return centroids_.size();
    } /*!< Number of centroids (next coarser level size). */

private:
    std::vector<Point> datas_;
    std::vector<size_t> labels_;
    std::vector<Point> centroids_;

    std::unique_ptr<raft::resources> handle_;

    size_t N_; //!< Number of data points.
    size_t dim_; //!< Data dimensionality.
    size_t k_; //!< Number of clusters for the current run.

    std::unique_ptr<raft::device_matrix<double, int>> dataset_gpu_;
    std::unique_ptr<raft::device_matrix<double, int>> centroids_gpu_;
    std::unique_ptr<raft::device_vector<int, int>> labels_gpu_;

    void uploadDataToGPU(const double* data, size_t N, size_t dim, bool is_aos);
    void downloadResultsFromGPU();
    void convertToPointVector(const double* flat_data, size_t N, size_t dim, bool is_aos);
};

#endif // CUDA_KMEANS_H
