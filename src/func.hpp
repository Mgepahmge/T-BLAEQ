#pragma once

#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <stdexcept>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>

namespace Comp {

/// @brief Return true if |val| < 1e-6.
inline bool isZero(double val) {
    static constexpr double kEps = 1e-6;
    return std::fabs(val) < kEps;
}

/// @brief Return true if |val| < epsilon.
inline bool isZeroWithEps(double val, double epsilon) {
    return std::fabs(val) < epsilon;
}

/// @brief Return true if two vectors are component-wise equal within kEps.
inline bool isVectorEqual(const std::vector<double>& a,
                          const std::vector<double>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!isZero(a[i] - b[i])) return false;
    return true;
}

/// @brief Return true if two matrices are component-wise equal within kEps.
inline bool isMatrixEqual(const std::vector<std::vector<double>>& m1,
                          const std::vector<std::vector<double>>& m2) {
    if (m1.size() != m2.size()) return false;
    if (m1.empty()) return true;
    const size_t cols = m1[0].size();
    for (const auto& row : m2)
        if (row.size() != cols) return false;
    for (size_t i = 0; i < m1.size(); ++i)
        for (size_t j = 0; j < cols; ++j)
            if (!isZero(m1[i][j] - m2[i][j])) return false;
    return true;
}

} // namespace Comp

namespace RandomSelector {

/**
 * @brief Sample elements from @p raw with independent Bernoulli probability @p p.
 * @param p   Inclusion probability in [0, 1].
 * @param raw Source vector.
 * @return    New vector containing the sampled elements.
 */
inline std::vector<size_t> sampleWithProbability(
    double p, const std::vector<size_t>& raw)
{
    if (p < 0.0 || p > 1.0)
        throw std::invalid_argument("p must be in [0, 1]");

    std::vector<size_t> result;
    std::bernoulli_distribution dist(p);
    std::mt19937 rng{std::random_device{}()};

    result.reserve(raw.size());
    for (size_t elem : raw)
        if (dist(rng)) result.push_back(elem);
    return result;
}

// Legacy shim
inline std::vector<size_t>
generate_with_probability_random_size_t_vec(double p,
                                            const std::vector<size_t>& raw) {
    return sampleWithProbability(p, raw);
}

} // namespace RandomSelector

namespace Simulation {

/**
 * @brief Print a progress bar to stdout.
 *        Format: [====>      ] 40.0% (40/100)
 */
inline void showProgress(int current, int total) {
    if (total <= 0) return;
    const float  progress = static_cast<float>(current) / static_cast<float>(total);
    constexpr int kWidth  = 50;
    const int    pos      = static_cast<int>(kWidth * progress);

    std::cout << '\r' << '[';
    for (int i = 0; i < kWidth; ++i) {
        if      (i < pos)  std::cout << '=';
        else if (i == pos) std::cout << '>';
        else               std::cout << ' ';
    }
    std::cout << "] " << std::fixed << std::setprecision(1)
              << (progress * 100.0f) << "% "
              << '(' << current << '/' << total << ')' << std::endl;
}

} // namespace Simulation

namespace Sort {

/**
 * @brief Return an index permutation that sorts @p labels in ascending or
 *        descending order.
 *
 *        After the call, labels[result[i]] is sorted.
 *
 * @param labels     Label vector to sort by.
 * @param ascending  Sort direction (default: ascending).
 * @return           Permutation array of the same length as labels.
 */
inline std::vector<size_t>
Sorted_Layer_With_Original_idxs(const std::vector<size_t>& labels,
                                 bool ascending = true)
{
    assert(!labels.empty());
    std::vector<size_t> idx(labels.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&labels, ascending](size_t i, size_t j) {
                  return ascending ? labels[i] < labels[j]
                                   : labels[i] > labels[j];
              });
    return idx;
}

} // namespace Sort

namespace Chrono {

/**
 * @brief Print the elapsed time between t1 and t2 with adaptive units:
 *         >= 1 s   -> "X.XXX s"
 *         >= 1 ms  -> "X.XXX ms"
 *         otherwise-> "X.XXX us"
 */
template<class Clock = std::chrono::steady_clock>
void printElapsed(const std::string& label,
                  const typename Clock::time_point& t1,
                  const typename Clock::time_point& t2)
{
    using namespace std::chrono;
    const auto dur = t2 - t1;

    std::cout << label << " : ";
    if      (dur >= duration<double>(1))
        std::cout << std::fixed << std::setprecision(3)
                  << duration<double>(dur).count() << " s\n";
    else if (dur >= duration<double, std::milli>(1))
        std::cout << std::fixed << std::setprecision(3)
                  << duration<double, std::milli>(dur).count() << " ms\n";
    else
        std::cout << std::fixed << std::setprecision(3)
                  << duration<double, std::micro>(dur).count() << " us\n";
}

/**
 * @brief Print the average elapsed time over @p totalTimes repetitions.
 */
template<class Clock = std::chrono::steady_clock>
void printAvgElapsed(const std::string& label,
                     const typename Clock::time_point& t1,
                     const typename Clock::time_point& t2,
                     size_t totalTimes)
{
    using namespace std::chrono;
    if (totalTimes == 0) return;
    const auto dur = (t2 - t1) / totalTimes;

    std::cout << label << " : ";
    if      (dur >= duration<double>(1))
        std::cout << std::fixed << std::setprecision(3)
                  << duration<double>(dur).count() << " s\n";
    else if (dur >= duration<double, std::milli>(1))
        std::cout << std::fixed << std::setprecision(3)
                  << duration<double, std::milli>(dur).count() << " ms\n";
    else
        std::cout << std::fixed << std::setprecision(3)
                  << duration<double, std::micro>(dur).count() << " us\n";
}

} // namespace Chrono

namespace dist {

/**
 * @brief Euclidean distance between two equal-length vectors.
 * @throws std::invalid_argument if a.size() != b.size().
 */
inline double euclidean(const std::vector<double>& a,
                        const std::vector<double>& b)
{
    if (a.size() != b.size())
        throw std::invalid_argument("euclidean: size mismatch");
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

/// @brief Squared Euclidean distance (avoids the sqrt).
inline double euclideanSquared(const std::vector<double>& a,
                               const std::vector<double>& b)
{
    if (a.size() != b.size())
        throw std::invalid_argument("euclideanSquared: size mismatch");
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

// Legacy shims
inline double euclidean_squared(const std::vector<double>& a,
                                const std::vector<double>& b) {
    return euclideanSquared(a, b);
}

} // namespace dist
