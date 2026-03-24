/**
 * @file check.cuh
 * @brief CUDA error-checking macros used throughout the project.
 *
 * @details Provides CUDA_CHECK for wrapping any CUDA API call and
 * CHECK_MEM_POS for asserting the memory residency (host/device/managed)
 * of a pointer at a specific location in the code.
 */

#ifndef BLAEQ_CUDA_CHECK_CUH
#define BLAEQ_CUDA_CHECK_CUH

#include <iostream>

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess) {                                                                                      \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err)           \
                      << std::endl;                                                                                    \
            std::cerr << "Failed call: " << #call << std::endl;                                                        \
            std::exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                              \
    }                                                                                                                  \
    while (0)

inline std::string getCudaMemoryTypeName(cudaMemoryType type) {
    switch (type) {
    case cudaMemoryTypeHost:
        return "Host";
    case cudaMemoryTypeDevice:
        return "Device";
    case cudaMemoryTypeManaged:
        return "Managed";
    case cudaMemoryTypeUnregistered:
        return "Unregistered";
    default:
        return "Unknown";
    }
}

#define CHECK_MEM_POS(data, expected)                                                                                  \
    do {                                                                                                               \
        cudaPointerAttributes attrib{};                                                                                \
        CUDA_CHECK(cudaPointerGetAttributes(&attrib, data));                                                           \
        if (attrib.type != (expected)) {                                                                               \
            std::cerr << "Memory position check failed at " << __FILE__ << ":" << __LINE__ << std::endl;               \
            std::cerr << "Expected memory type: " << getCudaMemoryTypeName(expected)                                   \
                      << ", but got: " << getCudaMemoryTypeName(attrib.type) << std::endl;                             \
            std::exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                              \
    }                                                                                                                  \
    while (0)

#endif // BLAEQ_CUDA_CHECK_CUH
