/**
 * @file CPUKmeans.cuh
 * @brief CPU fallback KMeans implementation for large datasets.
 *
 * @details Implements Lloyd's algorithm entirely on the host.  Used as a
 * fallback when the dataset is too large to fit in device memory for the
 * GPU KMeans (CUDAKmeans).  Build-index performance is not a priority, so
 * a straightforward implementation is preferred over an optimised one.
 *
 * The interface mirrors CUDAKmeans so that IndexBuilder can use either
 * implementation transparently.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

/**
 * @class CPUKmeans
 * @brief CPU-only KMeans using Lloyd's algorithm with KMeans++ initialisation.
 *
 * @details Drop-in replacement for CUDAKmeans when device memory is
 * insufficient to hold the dataset.  All computation runs on the host;
 * no CUDA calls are made.
 */
class CPUKmeans {
public:
    using Point = std::vector<double>;

    /*!
     * @brief Construct from a flat AOS host array.
     *
     * @param[in] data   Host pointer to the dataset (N * dim doubles, AOS layout).
     * @param[in] N      Number of data points.
     * @param[in] dim    Data dimensionality.
     * @param[in] is_aos True when data is in AOS layout (currently only AOS supported).
     */
    CPUKmeans(const double* data, size_t N, size_t dim, bool is_aos = true) : N_(N), dim_(dim), k_(0) {
        datas_.reserve(N);
        labels_.resize(N, 0);
        if (is_aos) {
            for (size_t i = 0; i < N; ++i) {
                datas_.emplace_back(data + i * dim, data + i * dim + dim);
            }
        }
    }

    ~CPUKmeans() = default;

    /*!
     * @brief Run KMeans clustering into k clusters using Lloyd's algorithm.
     *
     * @details Initialises centroids with KMeans++ then iterates until
     * convergence or max_iters is reached.
     *
     * @param[in] k         Number of clusters.
     * @param[in] max_iters Maximum number of Lloyd iterations.
     */
    void run(size_t k, size_t max_iters = 100) {
        k_ = k;
        initKMeansPlusPlus(k);

        for (size_t iter = 0; iter < max_iters; ++iter) {
            bool changed = assignLabels();
            recomputeCentroids();
            if (!changed)
                break;
        }
    }

    /*!
     * @brief Promote centroids to the new dataset and reset cluster state.
     *
     * @details After reset(), datas_ contains the centroids from the previous
     * run().  A new run() call will cluster those centroids.
     */
    void reset() {
        datas_ = std::move(centroids_);
        N_ = datas_.size();
        labels_.assign(N_, 0);
        centroids_.clear();
        k_ = 0;
    }

    [[nodiscard]] const std::vector<size_t>& getLabels()     const { return labels_; }     /*!< Per-point cluster assignment. */
    [[nodiscard]] const std::vector<Point>&  getCentroids()  const { return centroids_; }  /*!< Centroid coordinates after run(). */
    [[nodiscard]] const std::vector<Point>&  getdatas()      const { return datas_; }      /*!< Current level dataset points. */
    [[nodiscard]] size_t get_curr_layer_length() const { return datas_.size(); }     /*!< Number of points at this level. */
    [[nodiscard]] size_t get_next_layer_length() const { return centroids_.size(); } /*!< Number of centroids produced. */

private:
    std::vector<Point>  datas_;
    std::vector<size_t> labels_;
    std::vector<Point>  centroids_;
    size_t N_;
    size_t dim_;
    size_t k_;

    /*!
     * @brief KMeans++ centroid initialisation.
     *
     * @details Selects centroids sequentially: the first is chosen uniformly
     * at random; each subsequent centroid is chosen with probability
     * proportional to the squared distance from the nearest already-chosen
     * centroid.  This gives O(log k) approximation to the optimal solution.
     */
    void initKMeansPlusPlus(size_t k) {
        centroids_.clear();
        centroids_.reserve(k);

        std::mt19937_64 rng(42);

        // First centroid: uniform random
        std::uniform_int_distribution<size_t> uni(0, N_ - 1);
        centroids_.push_back(datas_[uni(rng)]);

        std::vector<double> minDist(N_, std::numeric_limits<double>::max());

        for (size_t c = 1; c < k; ++c) {
            // Update min distances to nearest centroid so far
            const Point& last = centroids_.back();
            double total = 0.0;
            for (size_t i = 0; i < N_; ++i) {
                double d = squaredDist(datas_[i], last);
                if (d < minDist[i])
                    minDist[i] = d;
                total += minDist[i];
            }

            // Sample next centroid proportional to squared distance
            std::uniform_real_distribution<double> dist(0.0, total);
            double threshold = dist(rng);
            double cumsum = 0.0;
            size_t chosen = N_ - 1;
            for (size_t i = 0; i < N_; ++i) {
                cumsum += minDist[i];
                if (cumsum >= threshold) {
                    chosen = i;
                    break;
                }
            }
            centroids_.push_back(datas_[chosen]);
        }
    }

    /*!
     * @brief Assign each point to its nearest centroid.
     *
     * @return True if any label changed (convergence check).
     */
    bool assignLabels() {
        bool changed = false;
        for (size_t i = 0; i < N_; ++i) {
            double best = std::numeric_limits<double>::max();
            size_t bestC = 0;
            for (size_t c = 0; c < k_; ++c) {
                double d = squaredDist(datas_[i], centroids_[c]);
                if (d < best) {
                    best = d;
                    bestC = c;
                }
            }
            if (labels_[i] != bestC) {
                labels_[i] = bestC;
                changed = true;
            }
        }
        return changed;
    }

    /*!
     * @brief Recompute each centroid as the mean of its assigned points.
     *
     * @details Empty clusters retain their previous centroid position.
     */
    void recomputeCentroids() {
        std::vector<Point>  sums(k_, Point(dim_, 0.0));
        std::vector<size_t> counts(k_, 0);

        for (size_t i = 0; i < N_; ++i) {
            size_t c = labels_[i];
            ++counts[c];
            for (size_t d = 0; d < dim_; ++d)
                sums[c][d] += datas_[i][d];
        }

        for (size_t c = 0; c < k_; ++c) {
            if (counts[c] > 0) {
                for (size_t d = 0; d < dim_; ++d)
                    centroids_[c][d] = sums[c][d] / static_cast<double>(counts[c]);
            }
            // Empty cluster: keep previous centroid (no reinitialization needed
            // for index building purposes)
        }
    }

    static double squaredDist(const Point& a, const Point& b) {
        double s = 0.0;
        for (size_t d = 0; d < a.size(); ++d) {
            double diff = a[d] - b[d];
            s += diff * diff;
        }
        return s;
    }
};
