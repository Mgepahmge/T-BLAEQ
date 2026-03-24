#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
#include "L3Strategy.cuh"
#include "StrategyCommon.cuh"
#include "src/Kernel/SpMSpV.cuh"
#include "src/MergeSort/MergeSort.cuh"
#include "src/Query/GridCompact.cuh"
#include "src/Query/L3Kernels.cuh"
#include "src/Query/RangePruning.cuh"
#include "src/Query/Refactor.cuh"
#include "src/Query/compact.cuh"
#include "src/core/Memory.cuh"
#include "src/core/MemoryPolicy.cuh"
#include "src/utils/NVTXProfiler.cuh"

// Utility: available device memory with a safety margin
static size_t availableMem(double margin = 0.80) {
    size_t free = 0, total = 0;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return static_cast<size_t>(free * margin);
}

void L3Strategy::prepare(std::ostream& reportOs) {
    idx_.computeStats();
    PolicyScheduler::printReport(idx_, idx_.activePolicy, reportOs);
    // L3 uploads nothing in advance; all data is managed per-tile in runLevel().
    // Pre-size the device buffer vectors so per-level reset() calls work.
    idx_.dPTensorVals.resize(idx_.intervals);
    idx_.dMeshMaxRadius.resize(idx_.intervals);
}

LevelResult L3Strategy::runLevel(size_t l, SparseGrid* currentGrid, QueryType qType, const double* lo, const double* hi,
                                 const double* queryPoint, size_t K) {
    NvtxProfiler prof(("L3_Level" + std::to_string(l)).c_str(), NvtxProfiler::ColorMode::Fixed,
                      NvtxProfilerColor::Green);

    const size_t D = idx_.D;
    const size_t p = currentGrid->get_nnz_nums();

    // Step 0: ensure currentGrid is on host
    bool gridWasOnDevice = false;
    double* hCurVals = nullptr;
    size_t* hCurIds = nullptr;

    // Determine where currentGrid lives and get host pointers.
    //
    // Three cases:
    //   a) L3 produced the previous level's output -> vals_ is a host new[] ptr.
    //   b) L0/L1/L2 produced the previous level's output -> vals_ is a device ptr.
    //   c) Level 0: currentGrid is coarsestView built by QueryEngine using
    //      idx.dCoarsestMeshIds/Vals.  For L3, those device buffers were never
    //      uploaded (prepare() skips uploadPermanentData).  vals_ is nullptr.
    //      We fall back to the host coarsestMesh directly.
    if (currentGrid->get_vals_() == nullptr) {
        // Case (c): coarsestMesh not on device -- use host copy
        SparseGrid* hCoarsest = idx_.coarsestMesh;
        hCurVals = const_cast<double*>(hCoarsest->get_vals_());
        hCurIds = const_cast<size_t*>(hCoarsest->get_ids_());
    }
    else {
        cudaPointerAttributes attr{};
        cudaPointerGetAttributes(&attr, currentGrid->get_vals_());
        gridWasOnDevice = (attr.type == cudaMemoryTypeDevice);

        if (gridWasOnDevice) {
            // Case (b): device-resident grid from L0/L1/L2 -- download
            hCurVals = new double[p * D];
            hCurIds = new size_t[p];
            CUDA_CHECK(cudaMemcpy(hCurVals, currentGrid->get_vals_(), p * D * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hCurIds, currentGrid->get_ids_(), p * sizeof(size_t), cudaMemcpyDeviceToHost));
        }
        else {
            // Case (a): host-resident grid from previous L3 level
            hCurVals = const_cast<double*>(currentGrid->get_vals_());
            hCurIds = const_cast<size_t*>(currentGrid->get_ids_());
        }
    }

    // Upload radius for this level (small, always fits)
    const size_t rLen = idx_.meshSizes[idx_.intervals - l];
    idx_.dMeshMaxRadius[l].reset(idx_.meshMaxRadius[l], rLen);

    // Step 1: tiled pruning
    size_t selectCount = 0;
    bool* mask = runPruningTiled(l, hCurVals, hCurIds, p, qType, lo, hi, queryPoint, K, selectCount);

    // Step 2: tiled compact -> host pruned grid
    auto* prunedVals = new double[selectCount * D];
    auto* prunedIds = new size_t[selectCount];

    runCompactTiled(hCurVals, hCurIds, p, currentGrid->get_num_rows(), mask, selectCount, prunedVals, prunedIds, 0);
    delete[] mask;

    if (gridWasOnDevice) {
        delete[] hCurVals;
        delete[] hCurIds;
    }

    // Free transient radius
    idx_.dMeshMaxRadius[l].free();

    // Step 3: tiled SpTSpM
    double* yVals = nullptr;
    size_t* yIds = nullptr;
    size_t totalNnz = 0;

    runSpTSpMTiled(l, prunedVals, prunedIds, selectCount, yVals, yIds, totalNnz);

    delete[] prunedVals;
    delete[] prunedIds;

    // Step 4: tiled refactor
    runRefactorTiled(l, yIds, totalNnz);

    std::cout << "  Level " << l << " [L3] -> " << totalNnz << " pts\n";

    // Output grid is host-resident; caller frees with delete[] (ownsIds=true)
    SparseGrid* out = new SparseGrid{idx_.pTensors[l]->get_row_nums(), D, totalNnz, yIds, yVals};
    return {out, /*ownsIds=*/true, /*onHost=*/true};
}

bool* L3Strategy::runPruningTiled(size_t l, const double* hVals, const size_t* hIds, size_t p, QueryType qType,
                                  const double* lo, const double* hi, const double* queryPoint, size_t K,
                                  size_t& outSelectCount) {
    if (qType == QueryType::RANGE) {
        return runRangePruningTiled(l, hVals, hIds, p, lo, hi, outSelectCount);
    }
    else {
        return runKnnPruningTiled(l, hVals, hIds, p, queryPoint, K, outSelectCount);
    }
}

bool* L3Strategy::runRangePruningTiled(size_t l, const double* hVals, const size_t* hIds, size_t p, const double* lo,
                                       const double* hi, size_t& outSelectCount) {
    const size_t D = idx_.D;
    const double* hRadius = idx_.meshMaxRadius[l];

    // Pre-gather radius indexed by global centroid ID
    auto* hRadiusGathered = new double[p];
    for (size_t i = 0; i < p; ++i) {
        hRadiusGathered[i] = hRadius[hIds[i]];
    }

    // Upload query box once
    double* dLo = nullptr;
    double* dHi = nullptr;
    CUDA_CHECK(cudaMalloc(&dLo, D * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dHi, D * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dLo, lo, D * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dHi, hi, D * sizeof(double), cudaMemcpyHostToDevice));

    bool* fullMask = new bool[p]();
    outSelectCount = 0;

    size_t rStart = 0;
    while (rStart < p) {
        // Per centroid: centroids D*8 + radius 8 + mask 1
        const size_t avail = availableMem();
        const size_t perCent = D * sizeof(double) + sizeof(double) + sizeof(bool);
        const size_t tileSize = std::max<size_t>(1, avail / perCent);
        const size_t rLen = std::min(tileSize, p - rStart);

        double* dTileVals = nullptr;
        double* dTileRadius = nullptr;
        bool* dTileMask = nullptr;
        CUDA_CHECK(cudaMalloc(&dTileVals, rLen * D * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dTileRadius, rLen * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dTileMask, rLen * sizeof(bool)));
        CUDA_CHECK(cudaMemcpy(dTileVals, hVals + rStart * D, rLen * D * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dTileRadius, hRadiusGathered + rStart, rLen * sizeof(double), cudaMemcpyHostToDevice));

        const dim3 block(256);
        const dim3 grid((rLen + 255) / 256);
        rangePruningKernelL3<<<grid, block>>>(dTileMask, dLo, dHi, dTileVals, dTileRadius, D, rLen);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(fullMask + rStart, dTileMask, rLen * sizeof(bool), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < rLen; ++i) {
            if (fullMask[rStart + i]) {
                ++outSelectCount;
            }
        }

        CUDA_CHECK(cudaFree(dTileVals));
        CUDA_CHECK(cudaFree(dTileRadius));
        CUDA_CHECK(cudaFree(dTileMask));
        rStart += rLen;
    }

    delete[] hRadiusGathered;
    CUDA_CHECK(cudaFree(dLo));
    CUDA_CHECK(cudaFree(dHi));
    return fullMask;
}

bool* L3Strategy::runKnnPruningTiled(size_t l, const double* hVals, const size_t* hIds, size_t p,
                                     const double* queryPoint, size_t K, size_t& outSelectCount) {
    const size_t D = idx_.D;
    const double* hRadius = idx_.meshMaxRadius[l];

    // Pre-gather radius and clusterSizes by global centroid ID.
    // Eliminates all indirect addressing on device.
    auto* hRadiusGathered = new double[p];
    auto* hClusterSizesGathered = new size_t[p];
    const size_t* hClusterSizes = idx_.pTensors[l]->get_nnz_per_col();
    for (size_t i = 0; i < p; ++i) {
        hRadiusGathered[i] = hRadius[hIds[i]];
        hClusterSizesGathered[i] = hClusterSizes[hIds[i]];
    }

    // Upload query point once
    double* dQueryPoint = nullptr;
    CUDA_CHECK(cudaMalloc(&dQueryPoint, D * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dQueryPoint, queryPoint, D * sizeof(double), cudaMemcpyHostToDevice));

    // Accumulate tile-sorted ClusterL3 results for final host merge.
    // Each tile: distance kernel -> blockMergeSortKernel (on device) -> download.
    // Final: hostSerialMergeSort merges the locally-sorted segments.
    constexpr int kIPB = 512; // items per block
    constexpr int kIPT = 2; // items per thread
    constexpr int kTPB = kIPB / kIPT; // threads per block = 256
    const ClusterL3 sentinel = {std::numeric_limits<double>::max(), 0};

    std::vector<ClusterL3> allClusters;
    allClusters.reserve(p);

    size_t rStart = 0;
    while (rStart < p) {
        // Per centroid device cost: centroids D*8 + radius 8 + ClusterL3 x2 (in+out)
        const size_t avail = availableMem();
        const size_t perCent = D * sizeof(double) + sizeof(double) + 2 * sizeof(ClusterL3);
        const size_t tileSize = std::max<size_t>(1, avail / perCent);
        const size_t rLen = std::min(tileSize, p - rStart);

        double* dTileVals = nullptr;
        double* dTileRadius = nullptr;
        ClusterL3* dClusters = nullptr; // distance kernel output
        ClusterL3* dSorted = nullptr; // sort kernel output
        CUDA_CHECK(cudaMalloc(&dTileVals, rLen * D * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dTileRadius, rLen * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dClusters, rLen * sizeof(ClusterL3)));
        CUDA_CHECK(cudaMalloc(&dSorted, rLen * sizeof(ClusterL3)));

        CUDA_CHECK(cudaMemcpy(dTileVals, hVals + rStart * D, rLen * D * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dTileRadius, hRadiusGathered + rStart, rLen * sizeof(double), cudaMemcpyHostToDevice));

        // Step A: compute distances on device
        {
            const dim3 block(256);
            const dim3 grid((rLen + 255) / 256);
            calculateClusterDistanceKernelL3<<<grid, block>>>(dClusters, dQueryPoint, dTileVals, dTileRadius, D, rLen,
                                                              rStart);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Step B: block-sort on
        {
            const int blocks = (static_cast<int>(rLen) + kIPB - 1) / kIPB;
            blockMergeSortKernel<kIPB, kIPT, ClusterL3, true>
                <<<blocks, kTPB>>>(dClusters, dSorted, static_cast<int>(rLen), sentinel);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Step C: download sorted tile and append to allClusters
        const size_t prevSz = allClusters.size();
        allClusters.resize(prevSz + rLen);
        CUDA_CHECK(cudaMemcpy(allClusters.data() + prevSz, dSorted, rLen * sizeof(ClusterL3), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(dTileVals));
        CUDA_CHECK(cudaFree(dTileRadius));
        CUDA_CHECK(cudaFree(dClusters));
        CUDA_CHECK(cudaFree(dSorted));
        rStart += rLen;
    }

    CUDA_CHECK(cudaFree(dQueryPoint));

    // Host merge: allClusters consists of kIPB-sorted segments.
    hostSerialMergeSort<ClusterL3, true>(allClusters.data(), static_cast<int>(allClusters.size()), kIPB);

    bool* fullMask = new bool[p]();
    size_t currentCount = 0;
    double currentDistance = 0.0;
    outSelectCount = 0;

    for (const auto& cl : allClusters) {
        const size_t globalIdx = static_cast<size_t>(cl.label);
        if (currentCount >= K) {
            if (cl.distance > currentDistance) {
                break;
            }
            currentCount += hClusterSizesGathered[globalIdx];
            fullMask[globalIdx] = true;
            ++outSelectCount;
        }
        else {
            currentCount += hClusterSizesGathered[globalIdx];
            currentDistance = std::max(currentDistance, cl.distance + 2.0 * hRadiusGathered[globalIdx]);
            fullMask[globalIdx] = true;
            ++outSelectCount;
        }
    }

    delete[] hRadiusGathered;
    delete[] hClusterSizesGathered;
    return fullMask;
}


size_t L3Strategy::runCompactTiled(const double* hVals, const size_t* hIds, size_t p, size_t numRows, const bool* mask,
                                   size_t selectCount, double* outVals, size_t* outIds, size_t offset) {
    const size_t D = idx_.D;
    size_t written = 0;
    (void)numRows; // no longer needed: we call kernels directly

    // Constants matching GridCompact.cu
    constexpr int kBlockSize = 512;
    constexpr int kKp = 128;
    constexpr int kIter = kKp >> 5; // 4
    constexpr int kWarpPerBlock = kBlockSize >> 5; // 16

    size_t rStart = 0;
    while (rStart < p) {
        // Device memory per element (input + procCounts worst-case + output):
        // vals: D * 8B
        // ids: 8B
        // mask: 1B
        // procCounts: 4B  (1 uint per kKp elements, amortized)
        // out_vals: D * 8B  (worst case all selected)
        // out_ids: 8B
        const size_t avail = availableMem();
        const size_t perElem = 2 * D * sizeof(double) + 2 * sizeof(size_t) + sizeof(bool) + sizeof(unsigned int);
        const size_t tileSize = std::max<size_t>(kKp, avail / perElem);
        const size_t rLen = std::min(tileSize, p - rStart);

        // Count selected in this tile (host side, O(rLen))
        size_t tileSel = 0;
        for (size_t i = 0; i < rLen; ++i) {
            if (mask[rStart + i]) {
                ++tileSel;
            }
        }
        if (tileSel == 0) {
            rStart += rLen;
            continue;
        }

        // Upload tile
        double* dVals = nullptr;
        size_t* dIds = nullptr;
        bool* dMask = nullptr;
        CUDA_CHECK(cudaMalloc(&dVals, rLen * D * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dIds, rLen * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&dMask, rLen * sizeof(bool)));
        CUDA_CHECK(cudaMemcpy(dVals, hVals + rStart * D, rLen * D * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dIds, hIds + rStart, rLen * sizeof(size_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dMask, mask + rStart, rLen * sizeof(bool), cudaMemcpyHostToDevice));

        // Step A: count selected per warp
        const size_t nProc = (rLen + kKp - 1) / kKp;
        unsigned int* dProcCounts = nullptr;
        CUDA_CHECK(cudaMalloc(&dProcCounts, nProc * sizeof(unsigned int)));
        launchCountKernel<bool, kKp, kBlockSize>(dProcCounts, dMask, static_cast<unsigned int>(rLen));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Step B: prefix sum over procCounts
        launchPrefixSumKernel<unsigned int, kBlockSize>(dProcCounts, static_cast<unsigned int>(nProc));
        // (launchPrefixSumKernel synchronises internally)

        // Step C: scatter vals and ids
        double* dOutVals = nullptr;
        size_t* dOutIds = nullptr;
        CUDA_CHECK(cudaMalloc(&dOutVals, tileSel * D * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dOutIds, tileSel * sizeof(size_t)));

        const size_t gridSz = (nProc + kWarpPerBlock - 1) / kWarpPerBlock;
        gridCompactPrefixKernel<kIter><<<gridSz, kBlockSize>>>(dOutVals, dOutIds, dVals, dIds, dMask, dProcCounts,
                                                               static_cast<unsigned int>(rLen), D, nProc, tileSel);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Download and append to output
        CUDA_CHECK(cudaMemcpy(outVals + (offset + written) * D, dOutVals, tileSel * D * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(outIds + (offset + written), dOutIds, tileSel * sizeof(size_t), cudaMemcpyDeviceToHost));

        // Cleanup
        CUDA_CHECK(cudaFree(dVals));
        CUDA_CHECK(cudaFree(dIds));
        CUDA_CHECK(cudaFree(dMask));
        CUDA_CHECK(cudaFree(dProcCounts));
        CUDA_CHECK(cudaFree(dOutVals));
        CUDA_CHECK(cudaFree(dOutIds));

        written += tileSel;
        rStart += rLen;
    }

    return written;
}


void L3Strategy::runSpTSpMTiled(size_t l, const double* prunedVals, const size_t* prunedIds, size_t pruneCount,
                                double*& outYVals, size_t*& outYIds, size_t& outTotalNnz) {
    SparseTensorCscFormat* P = idx_.pTensors[l];
    const size_t D = idx_.D;
    const size_t* h_colRes = P->get_col_res();
    const size_t* h_rowIds = P->get_row_ids();
    const double* h_pVals = P->get_vals();

    // Sort pruned centroids by column index
    std::vector<size_t> sortPerm(pruneCount);
    std::iota(sortPerm.begin(), sortPerm.end(), 0);
    std::sort(sortPerm.begin(), sortPerm.end(), [&](size_t a, size_t b) { return prunedIds[a] < prunedIds[b]; });

    // Compute total output nnz
    size_t totalNnz = 0;
    for (size_t i = 0; i < pruneCount; ++i) {
        const size_t col = prunedIds[sortPerm[i]];
        totalNnz += h_colRes[col + 1] - h_colRes[col];
    }

    outYVals = new double[totalNnz * D];
    outYIds = new size_t[totalNnz];
    outTotalNnz = totalNnz;

    if (totalNnz == 0) {
        return;
    }

    size_t selIdx = 0;
    size_t outOff = 0;

    while (selIdx < pruneCount) {
        // Tile budget: per fine-point device cost:
        //   P.vals:   D * 8  (one nnz entry)
        //   gridVals: amortized over avg_nnz_per_col; conservatively D * 8
        //   colInd:   8
        //   matPos:   8
        //   yVals:    D * 8
        //   yIds:     8
        // Total per fine-point ~ (3D + 3) * 8
        const size_t avail = availableMem();
        const size_t perFP = (3 * D + 3) * sizeof(size_t);
        const size_t kMax = std::max<size_t>(1, avail / perFP);

        // Accumulate columns into tile until fine-point budget exhausted
        size_t tileSelStart = selIdx;
        size_t tileK = 0;
        while (selIdx < pruneCount) {
            const size_t col = prunedIds[sortPerm[selIdx]];
            const size_t colNnz = h_colRes[col + 1] - h_colRes[col];
            if (tileK + colNnz > kMax && tileK > 0) {
                break;
            }
            tileK += colNnz;
            ++selIdx;
        }
        // Edge case: single column exceeds budget -- must still make progress
        if (tileK == 0 && selIdx < pruneCount) {
            const size_t col = prunedIds[sortPerm[selIdx]];
            tileK = h_colRes[col + 1] - h_colRes[col];
            ++selIdx;
        }
        if (tileK == 0) {
            break;
        }

        const size_t tileSel = selIdx - tileSelStart;

        // Build host staging arrays.
        auto* hPV = new double[tileK * D];
        auto* hCI = new size_t[tileK];
        auto* hMP = new size_t[tileK];
        auto* hRI = new size_t[tileK];
        auto* hGV = new double[tileSel * D];

        size_t writeIdx = 0;
        size_t localSelIdx = 0;
        for (size_t si = tileSelStart; si < tileSelStart + tileSel; ++si) {
            const size_t origIdx = sortPerm[si];
            const size_t col = prunedIds[origIdx];
            const size_t colStart = h_colRes[col];
            const size_t colEnd = h_colRes[col + 1];
            const size_t colNnz = colEnd - colStart;

            // gridVals: one memcpy per centroid
            std::memcpy(hGV + localSelIdx * D, prunedVals + origIdx * D, D * sizeof(double));

            // P.vals: one memcpy per column block
            std::memcpy(hPV + writeIdx * D, h_pVals + colStart * D, colNnz * D * sizeof(double));

            // Index arrays: one pass over column entries
            for (size_t j = 0; j < colNnz; ++j) {
                hCI[writeIdx + j] = localSelIdx;
                hMP[writeIdx + j] = writeIdx + j;
                hRI[writeIdx + j] = h_rowIds[colStart + j];
            }

            writeIdx += colNnz;
            ++localSelIdx;
        }

        // Upload to device
        double* dPV = nullptr;
        size_t* dCI = nullptr;
        size_t* dMP = nullptr;
        double* dGV = nullptr;
        double* dYV = nullptr;
        size_t* dYI = nullptr;
        CUDA_CHECK(cudaMalloc(&dPV, tileK * D * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dCI, tileK * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&dMP, tileK * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&dGV, tileSel * D * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dYV, tileK * D * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dYI, tileK * sizeof(size_t)));

        CUDA_CHECK(cudaMemcpy(dPV, hPV, tileK * D * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dCI, hCI, tileK * sizeof(size_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dMP, hMP, tileK * sizeof(size_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dGV, hGV, tileSel * D * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dYI, hRI, tileK * sizeof(size_t), cudaMemcpyHostToDevice));

        // Launch kernel
        const auto totalElems = static_cast<unsigned int>(tileK * D);
        const dim3 block(512);
        const dim3 grid_dim((totalElems + UNROLL_FACTOR * 512 - 1) / (UNROLL_FACTOR * 512));
        SpMSpVKernelAOS_v2<<<grid_dim, block>>>(dYV, dCI, dMP, dPV, dGV, static_cast<unsigned int>(D), totalElems);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Download and append to output
        CUDA_CHECK(cudaMemcpy(outYVals + outOff * D, dYV, tileK * D * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(outYIds + outOff, dYI, tileK * sizeof(size_t), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(dPV));
        CUDA_CHECK(cudaFree(dCI));
        CUDA_CHECK(cudaFree(dMP));
        CUDA_CHECK(cudaFree(dGV));
        CUDA_CHECK(cudaFree(dYV));
        CUDA_CHECK(cudaFree(dYI));
        delete[] hPV;
        delete[] hCI;
        delete[] hMP;
        delete[] hRI;
        delete[] hGV;

        outOff += tileK;
    }
}

// Step 4: tiled refactor

void L3Strategy::runRefactorTiled(size_t l, size_t* yIds, size_t totalNnz) {
    if (totalNnz == 0) {
        return;
    }

    // maps[l+1] is always available as a host array; direct remap, no device.
    const size_t* hMap = idx_.maps[l + 1];
    for (size_t i = 0; i < totalNnz; ++i) {
        yIds[i] = hMap[yIds[i]];
    }
}
