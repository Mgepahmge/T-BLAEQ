//
// Created by mingyu on 2026/3/16.
//

#include "Utils.cuh"

std::string extractDatasetName(const std::string& path) {
    size_t start = path.find_last_of('/') + 1;
    size_t end = path.find('_', start);
    return path.substr(start, end - start);
}

int extractRangePercentage(const std::string& path) {
    size_t pos = path.find_last_of('_') + 1;
    size_t dotPos = path.find('.', pos);
    return std::stoi(path.substr(pos, dotPos - pos));
}

std::string extractRangeInfo(const std::string& path) {
    size_t pos = path.find_last_of('_') + 1;
    size_t dotPos = path.find('.', pos);
    std::string rangeStr = path.substr(pos, dotPos - pos);

    size_t dashPos = rangeStr.find('-');
    if (dashPos != std::string::npos) {
        int percentage = std::stoi(rangeStr.substr(0, dashPos));
        int dimensions = std::stoi(rangeStr.substr(dashPos + 1));

        // Calc real range
        double actualCoverage = std::pow(percentage / 100.0, dimensions) * 100.0;

        std::ostringstream oss;
        oss << percentage << "%^" << dimensions << "="
            << std::fixed << std::setprecision(2) << actualCoverage << "%";
        return oss.str();
    }

    // If no '-' found, return original format
    return rangeStr + "%";
}