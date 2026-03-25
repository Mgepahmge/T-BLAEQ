#include "Setup.cuh"
#include "src/func.hpp"

size_t Compute_Layer_nums(size_t /*N*/) { return 4; }

size_t Compute_Centroid_nums(size_t dataNums, size_t ratio) {
    if (ratio == 0 || dataNums / ratio == 0) {
        return 1;
    }
    return dataNums / ratio;
}


std::string getQueryTypeString(QueryType qType) {
    switch (qType) {
    case QueryType::RANGE:
        return "RANGE";
    case QueryType::POINT:
        return "POINT";
    default:
        return "UNKNOWN";
    }
}
