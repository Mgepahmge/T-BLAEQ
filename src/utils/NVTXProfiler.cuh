/**
 * @file NVTXProfiler.cuh
 * @brief RAII NVTX range wrapper for GPU performance profiling.
 *
 * @details NvtxProfiler pushes an NVTX range on construction and pops it
 * on destruction or when release() is called manually.  It supports fixed
 * colours, random colours, and automatic range-call counting so that
 * repeated calls to the same named range are distinguished by a suffix (#2, #3...).
 * Used throughout the strategy and kernel code to annotate timeline regions
 * visible in Nsight Systems.
 */

#ifndef BLAEQ_CUDA_NVTXPROFILER_CUH
#define BLAEQ_CUDA_NVTXPROFILER_CUH

#include <array>
#include <atomic>
#include <nvtx3/nvToolsExt.h>
#include <random>
#include <string>
#include <unordered_map>

/**
 * @enum NvtxProfilerColor::Color
 * @brief Predefined ARGB colours for NVTX range annotations.
 */
namespace NvtxProfilerColor {
    enum Color : uint32_t {
        Red = 0xFF0000FF,
        Green = 0xFF00FF00,
        Blue = 0xFFFF0000,
        Cyan = 0xFFFFFF00,
        Magenta = 0xFFFF00FF,
        Yellow = 0xFF00FFFF,
        Orange = 0xFFFF8000,
        Purple = 0xFF8000FF,
        SpringGreen = 0xFF00FF80,
        LimeGreen = 0xFF80FF00,
        Rose = 0xFFFF0080,
        SkyBlue = 0xFF0080FF
    };
}

/**
 * @class NvtxProfiler
 * @brief RAII NVTX range: pushes on construction, pops on destruction or release().
 *
 * @details Non-copyable; movable.  An internal static counter per name string
 * ensures that each call to the same region is numbered sequentially, making
 * it easy to distinguish multiple iterations in the Nsight timeline.
 */
class NvtxProfiler {
public:
    /**
     * @enum ColorMode
     * @brief Selects how the NVTX range colour is chosen.
     */
    enum class ColorMode { None, Fixed, Random };

private:
    bool active_ = false;

    static std::unordered_map<std::string, std::atomic<uint32_t>>& getCounters() {
        static std::unordered_map<std::string, std::atomic<uint32_t>> counters;
        return counters;
    }

    static constexpr std::array<uint32_t, 12> COLOR_PALETTE = {0xFF0000FF, 0xFF00FF00, 0xFFFF0000, 0xFFFFFF00,
                                                               0xFFFF00FF, 0xFF00FFFF, 0xFFFF8000, 0xFF8000FF,
                                                               0xFF00FF80, 0xFF80FF00, 0xFFFF0080, 0xFF0080FF};

    static uint32_t getRandomColor() {
        thread_local std::mt19937 gen(std::random_device{}());
        thread_local std::uniform_int_distribution<size_t> dist(0, COLOR_PALETTE.size() - 1);
        return COLOR_PALETTE[dist(gen)];
    }

public:
    /*!
     * @brief Push an NVTX range with the given name and colour.
     *
     * @param[in] name       Human-readable range label.
     * @param[in] colorMode  Colour selection mode.
     * @param[in] fixedColor ARGB colour used when colorMode == Fixed.
     */
    explicit NvtxProfiler(const char* name, const ColorMode colorMode = ColorMode::None,
                          const uint32_t fixedColor = 0xFFFFFFFF) : active_(true) {

        const auto count = ++getCounters()[name];

        char buffer[256];
        if (count > 1) {
            snprintf(buffer, sizeof(buffer), "%s#%u", name, count);
        }
        else {
            snprintf(buffer, sizeof(buffer), "%s", name);
        }

        nvtxEventAttributes_t attrib = {0};
        attrib.version = NVTX_VERSION;
        attrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
        attrib.message.ascii = buffer;

        if (colorMode != ColorMode::None) {
            attrib.colorType = NVTX_COLOR_ARGB;
            attrib.color = (colorMode == ColorMode::Random) ? getRandomColor() : fixedColor;
        }

        nvtxRangePushEx(&attrib);
    }

    NvtxProfiler(const NvtxProfiler&) = delete;
    NvtxProfiler& operator=(const NvtxProfiler&) = delete;

    NvtxProfiler(NvtxProfiler&& other) noexcept : active_(other.active_) { other.active_ = false; }

    NvtxProfiler& operator=(NvtxProfiler&& other) noexcept {
        if (this != &other) {
            release();
            active_ = other.active_;
            other.active_ = false;
        }
        return *this;
    }

    /*!
     * @brief Pop the NVTX range immediately, before the destructor runs.
     *
     * @details Useful when the range should end before the enclosing scope exits.
     * Subsequent calls to release() or the destructor are no-ops.
     */
    void release() {
        if (active_) {
            nvtxRangePop();
            active_ = false;
        }
    }

    ~NvtxProfiler() { release(); }

    bool isActive() const { return active_; } //!< True while the NVTX range is still open.
};

#define NVTX_PROFILE(name) NvtxProfiler nvtx_profiler_##__LINE__(name)

#define NVTX_PROFILE_COLOR(name, color)                                                                                \
    NvtxProfiler nvtx_profiler_##__LINE__(name, NvtxProfiler::ColorMode::Fixed, color)

#define NVTX_PROFILE_RANDOM(name) NvtxProfiler nvtx_profiler_##__LINE__(name, NvtxProfiler::ColorMode::Random)

#endif // BLAEQ_CUDA_NVTXPROFILER_CUH
