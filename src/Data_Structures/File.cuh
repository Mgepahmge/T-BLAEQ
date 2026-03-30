/**
 * @file File.cuh
 * @brief Dataset and query file I/O, QueryType enum, and Query batch container.
 *
 * @details This file provides functions for loading and saving datasets and
 * query batches in the T-BLAEQ text format, the QueryType enum that
 * discriminates between range and KNN queries, and the Query class that
 * holds a batch of queries loaded from disk.
 */

#pragma once

#include "src/Data_Structures/Data_Structures.cuh"

/*!
 * @brief Load a dataset from a text file.
 *
 * @details File format: first line is "N D" (point count and dimensionality);
 * subsequent lines each contain D space-separated doubles for one point.
 *
 * @param[in] filename Path to the dataset file.
 * @return PointCloud owning the loaded data.
 * @throws std::runtime_error On file open or format errors.
 */
PointCloud loadFromFile(const std::string& filename);

/*!
 * @brief Save a PointCloud to a text file in the same format as loadFromFile.
 *
 * @param[in] cloud    The point cloud to save.
 * @param[in] filename Output file path.
 */
void saveToFile(const PointCloud& cloud, const std::string& filename);

/**
 * @enum QueryType
 * @brief Discriminates between the two supported query modes.
 */
enum class QueryType {
    POINT, //!< KNN query: find the K nearest neighbours of a query point.
    RANGE //!< Range query: find all points within an axis-aligned box.
};

/**
 * @class Query
 * @brief In-memory representation of a batch of queries loaded from a file.
 *
 * @details For POINT queries, data stores length*D doubles (one row per point).
 * For RANGE queries, data stores length*2*D doubles: each query occupies two
 * consecutive rows -- the lower bound row followed by the upper bound row.
 */
class Query {
public:
    int length = 0; //!< Number of queries in the batch.
    int dim = 0; //!< Dimensionality of each query.
    QueryType type = QueryType::POINT; //!< POINT or RANGE.
    std::vector<double> data; //!< Flat storage for all query values.
    std::string queryRangeInfo; //!< Query range information.

    /*!
     * @brief Construct an empty query batch and allocate storage.
     *
     * @param[in] len Number of queries.
     * @param[in] d   Dimensionality of each query.
     * @param[in] t   Query type (POINT or RANGE).
     */
    Query(int len, int d, QueryType t = QueryType::POINT) : length(len), dim(d), type(t) {
        data.resize(t == QueryType::POINT ? static_cast<size_t>(len) * d : static_cast<size_t>(len) * 2 * d);
    }

    /*!
     * @brief Return the query point at position index as a vector.
     *
     * @param[in] index Zero-based query index.
     * @return D-dimensional query point coordinates.
     * @throws std::runtime_error When type != POINT.
     * @throws std::out_of_range  When index >= length.
     */
    std::vector<double> getQueryPoint(int index) const;

    /*!
     * @brief Return the lower/upper bound pair for the range query at index.
     *
     * @param[in] index Zero-based query index.
     * @return Pair of D-dimensional lower and upper bound vectors.
     * @throws std::runtime_error When type != RANGE.
     * @throws std::out_of_range  When index >= length.
     */
    std::pair<std::vector<double>, std::vector<double>> getQueryRange(int index) const;

    /*!
     * @brief Set the query point at index.
     *
     * @param[in] index Zero-based query index.
     * @param[in] point D-dimensional point coordinates.
     * @throws std::runtime_error When type != POINT.
     */
    void setQueryPoint(int index, const std::vector<double>& point);

    /*!
     * @brief Set the lower and upper bounds for the range query at index.
     *
     * @param[in] index Zero-based query index.
     * @param[in] lower D-dimensional lower bound vector.
     * @param[in] upper D-dimensional upper bound vector.
     * @throws std::runtime_error When type != RANGE.
     */
    void setQueryRange(int index, const std::vector<double>& lower, const std::vector<double>& upper);
};


/*!
 * @brief Load a POINT query batch from file.
 *
 * @details File format: first line is "N D"; subsequent lines are query points.
 *
 * @param[in] filename Path to the query file.
 * @return Query batch of type POINT.
 */
Query loadQueryPointFromFile(const std::string& filename);

/*!
 * @brief Save a POINT query batch to file.
 *
 * @param[in] query    The query batch to save.
 * @param[in] filename Output file path.
 */
void saveQueryPointToFile(const Query& query, const std::string& filename);

/*!
 * @brief Load a RANGE query batch from file.
 *
 * @details File format: first line is "N D"; subsequent lines alternate between
 * lower-bound rows and upper-bound rows (two rows per query).
 *
 * @param[in] filename Path to the query file.
 * @return Query batch of type RANGE.
 */
Query loadQueryRangeFromFile(const std::string& filename);

/*!
 * @brief Save a RANGE query batch to file.
 *
 * @param[in] query    The query batch to save.
 * @param[in] filename Output file path.
 */
void saveQueryRangeToFile(const Query& query, const std::string& filename);
