//
// Created by mingyu on 2026/3/16.
//

#ifndef T_BLAEQ_UTILS_CUH
#define T_BLAEQ_UTILS_CUH
#include <iomanip>

// Extract dataset name（example: "gist", "sift"）
std::string extractDatasetName(const std::string& path);

// Extract range percentage（example: 10, 30, 50...）
int extractRangePercentage(const std::string& path);

std::string extractRangeInfo(const std::string& path);

#endif //T_BLAEQ_UTILS_CUH