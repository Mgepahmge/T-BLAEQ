/**
 * @file IndexData.cuh
 * @brief Central data container for the T-BLAEQ hierarchical mesh index.
 *
 * @details This file declares IndexData, the aggregate that owns every host
 * and device buffer required by the index.  It also declares MemoryStats,
 * a snapshot of memory consumption used by the policy scheduler.
 *
 * IndexData is a pure data container: it performs no query or build logic.
 * Strategy classes (L0-L3) call the upload helpers declared here to move
 * data to the device at the time appropriate for their policy.
 *
 * Memory layout:
 *   Host (always resident): pTensors, meshMaxRadius, maps, coarsestMesh.
 *   Device permanent (uploaded once by uploadPermanentData()): dMaps,
 *     dCoarsestMeshIds, dCoarsestMeshVals.
 *   Device policy-dependent: dPTensorVals and dMeshMaxRadius are populated
 *     permanently by L0/L1, and left empty by L2/L3 which manage them
 *     transiently inside runLevel().
 *   Device L0-only working storage: l0Bufs, allocated by allocL0WorkBuffers()
 *     and reused across all queries to achieve zero cudaMalloc in the hot path.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>
#include "Memory.cuh"
#include "MemoryPolicy.cuh"
#include "src/Data_Structures/Data_Structures.cuh"

/**
 * @struct MemoryStats
 * @brief Snapshot of host and device memory consumption for the loaded index.
 *
 * @details Computed once by IndexData::computeStats() during prepare().
 * The device fields reflect the L1 baseline (all data permanently resident).
 * deviceSpTSpMBufs is the additional cost incurred when all levels use L0.
 */
struct MemoryStats {
    size_t hostPTensors = 0; //!< Host memory occupied by all P-tensor value arrays.
    size_t hostMaps = 0; //!< Host memory occupied by all sort-to-original maps.
    size_t hostRadius = 0; //!< Host memory occupied by all max-radius arrays.
    size_t hostCoarsestMesh = 0; //!< Host memory occupied by the coarsest mesh grid.
    size_t hostTotal = 0; //!< Sum of all host fields above.

    size_t devicePTensorVals = 0; //!< Device memory for P-tensor values (L1 baseline).
    size_t deviceMaps = 0; //!< Device memory for sort-to-original maps.
    size_t deviceRadius = 0; //!< Device memory for max-radius arrays.
    size_t deviceCoarsestMesh = 0; //!< Device memory for the coarsest mesh.
    size_t deviceSpTSpMBufs = 0; //!< Additional device cost when all levels are L0.
    size_t deviceTotal = 0; //!< Sum of device fields above (excludes deviceSpTSpMBufs).

    size_t grandTotal = 0; //!< hostTotal + deviceTotal.

    /*!
     * @brief Print a human-readable memory statistics table to a stream.
     * @param[in] os Output stream.
     */
    void print(std::ostream& os) const;

    /*!
     * @brief Serialise memory statistics to a CSV row.
     * @return A comma-separated string of all fields.
     */
    std::string toCsv() const;
};

/**
 * @struct IndexData
 * @brief Owns all host and device buffers constituting the hierarchical mesh index.
 *
 * @details IndexData is a non-copyable aggregate that stores:
 *   - Structural parameters describing the dataset and index geometry.
 *   - Host-side index data: P-tensors, max-radius arrays, sort-to-original maps,
 *     and the coarsest mesh grid.
 *   - Device buffers whose lifetimes are controlled by the active strategy.
 *   - L0-specific working storage (l0Bufs) enabling zero-malloc query execution.
 *
 * Ownership rules:
 *   1. Raw cudaMalloc/cudaFree must not appear outside this struct's methods.
 *   2. Each IQueryStrategy calls uploadPermanentData() and optionally
 *      allocSpTSpMBuffers() or allocL0WorkBuffers() from its prepare() method.
 *   3. After prepare() completes, stats is valid and permanentDataOnDevice is true.
 */
struct IndexData {
    size_t D = 0; //!< Data dimensionality.
    size_t N = 0; //!< Total number of data points in the dataset.
    size_t height = 0; //!< Number of mesh levels (including the finest).
    size_t intervals = 0; //!< Number of P-tensor intervals (= height - 1).
    bool isAosArch = true; //!< True when values are stored in AoS layout.
    std::string datasetName; //!< Human-readable name of the dataset.

    std::vector<SparseTensorCscFormat*> pTensors; //!< [HOST] P-tensor per level in CSC format.
    std::vector<double*> meshMaxRadius; //!< [HOST] Max-radius array per level.
    std::vector<size_t> meshSizes; //!< Mesh point counts ordered finest to coarsest.
    std::vector<size_t> ratios; //!< Coarsening ratio per level.

    std::vector<size_t*> maps; //!< [HOST] Sort-to-original index maps per level.
    SparseGrid* coarsestMesh = nullptr; //!< [HOST] Coarsest mesh grid.

    std::vector<DeviceBuffer<size_t>> dMaps; //!< [DEVICE] Permanent sort-to-original maps.
    DeviceBuffer<size_t> dCoarsestMeshIds; //!< [DEVICE] Permanent coarsest mesh ids.
    DeviceBuffer<double> dCoarsestMeshVals; //!< [DEVICE] Permanent coarsest mesh vals.

    std::vector<DeviceBuffer<double>>
        dPTensorVals; //!< [DEVICE] P-tensor values per level (L0/L1 permanent; L2/L3 unused here).
    std::vector<DeviceBuffer<double>>
        dMeshMaxRadius; //!< [DEVICE] Max-radius per level (L0/L1 permanent; L2 transient).

    /**
     * @struct SpTSpMBuffers
     * @brief Pre-allocated device index buffers for L0's zero-malloc SpTSpM path.
     *
     * @details Each buffer is sized to P[l].row_nums (equal to P[l].nnz for this
     * index type) and allocated once by allocSpTSpMBuffers().  colInd and matrixPos
     * are overwritten each query; rowInd is reused as the output yIndex of the
     * returned SparseGrid (LevelResult::ownsIds = false).
     */
    struct SpTSpMBuffers {
        DeviceBuffer<size_t> colInd; //!< Local centroid index per output nnz entry.
        DeviceBuffer<size_t> rowInd; //!< Output fine-point id per nnz; reused as yIndex.
        DeviceBuffer<size_t> matrixPos; //!< Local nnz offset into the compact P-vals buffer.
    };
    std::vector<SpTSpMBuffers> dSpTSpMBufs; //!< [DEVICE] L0-only SpTSpM index buffers, one per level.

    /**
     * @struct L0WorkBuffers
     * @brief Query-time working storage enabling L0Strategy's zero-malloc hot path.
     *
     * @details All buffers are allocated once in allocL0WorkBuffers() using the
     * maximum required capacity across all levels.  They are reused for every
     * query and every level without any reallocation.
     *
     * Device buffers are sized to max(P[l].nnz) across all levels (max_nnz).
     * Host staging buffers are sized identically and reused across queries.
     */
    struct L0WorkBuffers {
        DeviceBuffer<bool> dMask; //!< Pruning output mask (max_nnz bools).
        DeviceBuffer<uint8_t> dClusters; //!< KNN distance results (max_nnz * 16 bytes).
        DeviceBuffer<double> dQueryPoint; //!< Uploaded query point for KNN pruning (D doubles).
        DeviceBuffer<double> dLo; //!< Uploaded lower query bound for RANGE pruning (D doubles).
        DeviceBuffer<double> dHi; //!< Uploaded upper query bound for RANGE pruning (D doubles).
        DeviceBuffer<unsigned int> dProcCounts; //!< Per-warp selection counts used by compact.
        DeviceBuffer<size_t> dCompactIds; //!< Compact output centroid ids (max_nnz entries).
        DeviceBuffer<double> dCompactVals; //!< Compact output centroid values (max_nnz * D doubles).
        std::vector<DeviceBuffer<double>> dYValue; //!< SpTSpM output values, one buffer per level.

        size_t* hVectorIndex = nullptr; //!< [HOST] Downloaded grid ids for SpTSpM staging (max_nnz).
        uint8_t* hClusters = nullptr; //!< [HOST] Sorted Cluster array plus bool-mask scratch area.
        size_t* hProcColInd = nullptr; //!< [HOST] Processed column indices for SpTSpM upload (max_nnz).
        size_t* hProcRowInd = nullptr; //!< [HOST] Processed row indices for SpTSpM upload (max_nnz).
        size_t* hProcMatrixPos = nullptr; //!< [HOST] Processed matrix positions for SpTSpM upload (max_nnz).

        bool ready = false; //!< True after allocL0WorkBuffers() has been called successfully.

        /*!
         * @brief Release all device and host buffers owned by this struct.
         */
        void release() {
            dMask.free();
            dClusters.free();
            dQueryPoint.free();
            dLo.free();
            dHi.free();
            dProcCounts.free();
            dCompactIds.free();
            dCompactVals.free();
            dYValue.clear();
            delete[] hVectorIndex;
            hVectorIndex = nullptr;
            delete[] hClusters;
            hClusters = nullptr;
            delete[] hProcColInd;
            hProcColInd = nullptr;
            delete[] hProcRowInd;
            hProcRowInd = nullptr;
            delete[] hProcMatrixPos;
            hProcMatrixPos = nullptr;
            ready = false;
        }
    };
    L0WorkBuffers l0Bufs; //!< L0-only working storage; empty until allocL0WorkBuffers() is called.

    IndexPolicy activePolicy; //!< Per-level memory policy selected by PolicyScheduler.

    bool permanentDataOnDevice = false; //!< True after uploadPermanentData() succeeds.
    bool spTSpMBufsReady = false; //!< True after allocSpTSpMBuffers() succeeds.

    MemoryStats stats; //!< Memory consumption snapshot; valid after computeStats().

    IndexData() = default;
    ~IndexData() { release(); }
    IndexData(const IndexData&) = delete;
    IndexData& operator=(const IndexData&) = delete;

    /*!
     * @brief Compute and cache host and device memory consumption statistics.
     *
     * @details Must be called before PolicyScheduler::recommend() or printReport().
     * Results are stored in the stats member.
     */
    void computeStats();

    /*!
     * @brief Release all host and device resources owned by this object.
     */
    void release();

    /*!
     * @brief Upload maps and coarsestMesh to the device as permanently-resident data.
     *
     * @details Called by every strategy's prepare().  Idempotent: a second call
     * is a no-op when permanentDataOnDevice is already true.
     */
    void uploadPermanentData();

    /*!
     * @brief Allocate pre-sized device index buffers for the L0 SpTSpM hot path.
     *
     * @details Sizes colInd, rowInd, and matrixPos for each L0-assigned level to
     * P[l].row_nums entries.  Must be called after uploadPermanentData().
     * Idempotent: a second call is a no-op when spTSpMBufsReady is true.
     */
    void allocSpTSpMBuffers();

    /*!
     * @brief Allocate all working buffers required by L0Strategy::runLevel().
     *
     * @details Sizes every buffer to max(P[l].nnz) across all levels so they can
     * be reused without reallocation for any level and any query.  Idempotent:
     * a second call is a no-op when l0Bufs.ready is true.
     */
    void allocL0WorkBuffers();

private:
    void releasePTensors(); /*!< Free all host P-tensor objects.*/
    void releaseMaps(); /*!< Free all host sort-to-original map arrays.*/
    void releaseRadius(); /*!< Free all host max-radius arrays.*/
};
