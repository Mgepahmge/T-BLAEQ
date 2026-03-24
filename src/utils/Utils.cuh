/**
 * @file Utils.cuh
 * @brief String parsing utilities for dataset and query file paths.
 *
 * @details Extracts structured information (dataset name, range percentage,
 * formatted range description) from file path conventions used throughout
 * the project for naming datasets and query files.
 */

#pragma once

#include <string>

/*!
 * @brief Extract the dataset name from a file path.
 *
 * @details Assumes the filename starts with the dataset name followed by an
 * underscore, e.g. "gist_960d_1M.txt" -> "gist".
 *
 * @param[in] path File path to parse.
 * @return Dataset name prefix before the first underscore.
 */
std::string extractDatasetName(const std::string& path);

/*!
 * @brief Extract the integer range percentage from a query file path.
 *
 * @details Expects a suffix of the form "_50.txt", returning 50.
 *
 * @param[in] path Query file path to parse.
 * @return Integer range percentage.
 */
int extractRangePercentage(const std::string& path);

/*!
 * @brief Extract a formatted range-query description from a file path.
 *
 * @details Handles both simple ("_50.txt" -> "50%") and compound forms
 * ("_50-10.txt" -> "50%^10=X.XX%").
 *
 * @param[in] path Query file path to parse.
 * @return Human-readable range description string.
 */
std::string extractRangeInfo(const std::string& path);
