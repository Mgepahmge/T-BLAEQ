/**
 * @file Memory.cuh
 * @brief RAII buffer types for device and host memory management.
 *
 * @details This file provides two lightweight RAII wrappers: DeviceBuffer<T>
 * for device-resident arrays and HostBuffer<T> for host-resident arrays.
 * Both types are non-copyable but movable, and they hide all raw
 * cudaMalloc/cudaFree and new[]/delete[] calls from the rest of the codebase.
 */

#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include "src/Query/check.cuh"

/**
 * @class DeviceBuffer
 * @brief RAII owner of a contiguous device-side array.
 *
 * @details Invariant: after any successful mutating operation, data() returns
 * either a valid device pointer or nullptr (when size == 0). The caller must
 * never call cudaMalloc or cudaFree directly on the managed storage.
 *
 * DeviceBuffer is non-copyable and movable.  It is used throughout IndexData
 * to hold all device-resident arrays, ensuring that device memory is released
 * exactly once even in the presence of exceptions or early returns.
 *
 * @tparam T Element type of the device array.
 */
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    /*!
     * @brief Allocate n uninitialised elements on the device.
     *
     * @param[in] n Number of elements to allocate. A value of 0 is valid and
     *              results in a null pointer with size 0.
     */
    explicit DeviceBuffer(size_t n) : size_(n) {
        if (n > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
        }
    }

    /*!
     * @brief Allocate n elements on the device and upload from a host array.
     *
     * @param[in] src Host pointer to at least n valid elements.
     * @param[in] n   Number of elements to allocate and upload.
     */
    DeviceBuffer(const T* src, size_t n) : size_(n) {
        if (n > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
            CUDA_CHECK(cudaMemcpy(ptr_, src, n * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    ~DeviceBuffer() { free(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), size_(o.size_) {
        o.ptr_ = nullptr;
        o.size_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            free();
            ptr_ = o.ptr_;
            size_ = o.size_;
            o.ptr_ = nullptr;
            o.size_ = 0;
        }
        return *this;
    }

    T* data() { return ptr_; } /*!< Device pointer for kernel arguments. */
    const T* data() const { return ptr_; } /*!< Const device pointer for kernel arguments. */

    size_t size() const { return size_; } /*!< Number of allocated elements. */
    bool empty() const { return size_ == 0; } /*!< True when no memory is allocated. */

    /*!
     * @brief Upload n elements from a host array into this buffer.
     *
     * @details The buffer must already be allocated with at least n elements.
     * @param[in] src Host pointer to at least n valid elements.
     * @param[in] n   Number of elements to upload.
     * @throws std::runtime_error When n exceeds the current capacity.
     */
    void uploadFrom(const T* src, size_t n) {
        if (n > size_) {
            throw std::runtime_error("DeviceBuffer::uploadFrom: n exceeds capacity");
        }
        CUDA_CHECK(cudaMemcpy(ptr_, src, n * sizeof(T), cudaMemcpyHostToDevice));
    }

    /*!
     * @brief Download n elements from this buffer into a host array.
     *
     * @param[out] dst Host pointer to a buffer of at least n elements.
     * @param[in]  n   Number of elements to download.
     * @throws std::runtime_error When n exceeds the current size.
     */
    void downloadTo(T* dst, size_t n) const {
        if (n > size_) {
            throw std::runtime_error("DeviceBuffer::downloadTo: n exceeds size");
        }
        CUDA_CHECK(cudaMemcpy(dst, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost));
    }

    /*!
     * @brief Reallocate to exactly n elements and upload from a host array.
     *
     * @details Frees the existing allocation first, then allocates n elements
     * and copies from src. Equivalent to free() followed by the two-argument
     * constructor.
     *
     * @param[in] src Host pointer to at least n valid elements.
     * @param[in] n   Number of elements to allocate and upload.
     */
    void reset(const T* src, size_t n) {
        free();
        size_ = n;
        if (n > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
            CUDA_CHECK(cudaMemcpy(ptr_, src, n * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    /*!
     * @brief Reallocate to exactly n uninitialised elements.
     *
     * @details Frees the existing allocation first. The new memory is not
     * initialised. Used by IndexData to pre-size buffers whose content will
     * be filled later by a kernel or cudaMemcpy.
     *
     * @param[in] n Number of elements to allocate.
     */
    void resize(size_t n) {
        free();
        size_ = n;
        if (n > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
        }
    }

    /*!
     * @brief Release device memory and reset to empty (ptr_ = nullptr, size_ = 0).
     */
    void free() {
        if (ptr_) {
            CUDA_CHECK(cudaFree(ptr_));
            ptr_ = nullptr;
        }
        size_ = 0;
    }

private:
    T* ptr_ = nullptr; //!< Raw device pointer; nullptr when empty.
    size_t size_ = 0; //!< Number of allocated elements.
};

/**
 * @class HostBuffer
 * @brief RAII owner of a contiguous host-side array (new[] / delete[]).
 *
 * @details Provides the same interface as DeviceBuffer but for host memory.
 * Used for staging buffers that bridge host and device operations.
 *
 * HostBuffer is non-copyable and movable.
 *
 * @tparam T Element type of the host array.
 */
template <typename T>
class HostBuffer {
public:
    HostBuffer() = default;

    /*!
     * @brief Allocate n uninitialised elements on the host.
     *
     * @param[in] n Number of elements to allocate.
     */
    explicit HostBuffer(size_t n) : size_(n) {
        if (n > 0) {
            ptr_ = new T[n];
        }
    }

    /*!
     * @brief Allocate n elements and copy from src.
     *
     * @param[in] src Source host pointer to at least n valid elements.
     * @param[in] n   Number of elements to allocate and copy.
     */
    HostBuffer(const T* src, size_t n) : size_(n) {
        if (n > 0) {
            ptr_ = new T[n];
            std::copy(src, src + n, ptr_);
        }
    }

    ~HostBuffer() { delete[] ptr_; }

    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;

    HostBuffer(HostBuffer&& o) noexcept : ptr_(o.ptr_), size_(o.size_) {
        o.ptr_ = nullptr;
        o.size_ = 0;
    }
    HostBuffer& operator=(HostBuffer&& o) noexcept {
        if (this != &o) {
            delete[] ptr_;
            ptr_ = o.ptr_;
            size_ = o.size_;
            o.ptr_ = nullptr;
            o.size_ = 0;
        }
        return *this;
    }

    T* data() { return ptr_; } /*!< Raw host pointer. */
    const T* data() const { return ptr_; } /*!< Const raw host pointer. */
    size_t size() const { return size_; } /*!< Number of allocated elements. */
    bool empty() const { return size_ == 0; } /*!< True when no memory is allocated. */

    /*!
     * @brief Download n elements from a device array into this buffer.
     *
     * @details The buffer must already be allocated with at least n elements.
     * @param[in] dSrc Device pointer to at least n valid elements.
     * @param[in] n    Number of elements to download.
     * @throws std::runtime_error When n exceeds the current capacity.
     */
    void downloadFrom(const T* dSrc, size_t n) {
        if (n > size_) {
            throw std::runtime_error("HostBuffer::downloadFrom: n exceeds capacity");
        }
        CUDA_CHECK(cudaMemcpy(ptr_, dSrc, n * sizeof(T), cudaMemcpyDeviceToHost));
    }

    /*!
     * @brief Release host memory and reset to empty (ptr_ = nullptr, size_ = 0).
     */
    void free() {
        delete[] ptr_;
        ptr_ = nullptr;
        size_ = 0;
    }

private:
    T* ptr_ = nullptr; //!< Raw host pointer; nullptr when empty.
    size_t size_ = 0; //!< Number of allocated elements.
};
