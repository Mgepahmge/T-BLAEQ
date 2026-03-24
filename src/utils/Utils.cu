#include <cmath>
#include <iomanip>
#include <sstream>
#include "Utils.cuh"

std::string extractDatasetName(const std::string& path) {
    const size_t start = path.find_last_of('/') + 1;
    const size_t end = path.find('_', start);
    return path.substr(start, end - start);
}

int extractRangePercentage(const std::string& path) {
    const size_t pos = path.find_last_of('_') + 1;
    const size_t dotPos = path.find('.', pos);
    return std::stoi(path.substr(pos, dotPos - pos));
}

std::string extractRangeInfo(const std::string& path) {
    const size_t pos = path.find_last_of('_') + 1;
    const size_t dotPos = path.find('.', pos);
    const std::string seg = path.substr(pos, dotPos - pos);

    const size_t dashPos = seg.find('-');
    if (dashPos != std::string::npos) {
        const int percentage = std::stoi(seg.substr(0, dashPos));
        const int dimensions = std::stoi(seg.substr(dashPos + 1));
        const double coverage = std::pow(percentage / 100.0, dimensions) * 100.0;

        std::ostringstream oss;
        oss << percentage << "%^" << dimensions << "=" << std::fixed << std::setprecision(2) << coverage << "%";
        return oss.str();
    }

    return seg + "%";
}
