#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include "L0Strategy.cuh"
#include "StrategyCommon.cuh"
#include "src/Kernel/SpMSpV.cuh"
#include "src/MergeSort/MergeSort.cuh"
#include "src/Query/GridCompact.cuh"
#include "src/Query/KNN.cuh"
#include "src/Query/RangePruning.cuh"
#include "src/Query/Refactor.cuh"
#include "src/Query/compact.cuh"
#include "src/core/MemoryPolicy.cuh"
#include "src/utils/NVTXProfiler.cuh"

// compact constants matching GridCompact.cu
static constexpr int kCompactBlockSize = 512;
static constexpr int kCompactKp = 128;
static constexpr int kCompactIter = kCompactKp >> 5; // 4
static constexpr int kCompactWPB = kCompactBlockSize >> 5; // 16


void L0Strategy::prepare(std::ostream& reportOs) {
    idx_.computeStats();
    PolicyScheduler::printReport(idx_, idx_.activePolicy, reportOs);
    idx_.uploadPermanentData();

    // Upload P vals and radius permanently only for levels assigned L0.
    // In a mixed-policy index, L1/L2/L3 levels manage their own data in
    // runLevel() and must not have their buffers pre-filled here.
    idx_.dPTensorVals.resize(idx_.intervals);
    idx_.dMeshMaxRadius.resize(idx_.intervals);
    for (size_t l = 0; l < idx_.intervals; ++l) {
        if (idx_.activePolicy.levels[l] != LevelPolicy::L0) {
            continue;
        }
        const size_t vLen = idx_.pTensors[l]->get_nnz_nums() * idx_.pTensors[l]->get_dim();
        idx_.dPTensorVals[l].reset(idx_.pTensors[l]->get_vals(), vLen);
        const size_t rLen = idx_.meshSizes[idx_.intervals - l];
        idx_.dMeshMaxRadius[l].reset(idx_.meshMaxRadius[l], rLen);
    }
    idx_.allocSpTSpMBuffers();

    // Allocate L0 work buffers in IndexData
    idx_.allocL0WorkBuffers();
}


LevelResult L0Strategy::runLevel(size_t l, SparseGrid* currentGrid, QueryType qType, const double* lo, const double* hi,
                                 const double* queryPoint, size_t K) {
    assert(idx_.permanentDataOnDevice);
    assert(idx_.spTSpMBufsReady);
    assert(idx_.l0Bufs.ready);

    NvtxProfiler lp(("L0_Level" + std::to_string(l)).c_str(), NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Green);

    const size_t D = idx_.D;
    const double* dRadius = idx_.dMeshMaxRadius[l].data();
    const double* dPVals = idx_.dPTensorVals[l].data();
    const size_t* dMap = idx_.dMaps[l + 1].data();
    assert(dRadius != nullptr && dPVals != nullptr);

    const size_t p = currentGrid->get_nnz_nums();

    // Convenience aliases into IndexData L0 buffers
    bool* dMask = idx_.l0Bufs.dMask.data();
    void* dClusters = idx_.l0Bufs.dClusters.data();
    double* dQueryPoint = idx_.l0Bufs.dQueryPoint.data();
    double* dLo = idx_.l0Bufs.dLo.data();
    double* dHi = idx_.l0Bufs.dHi.data();
    unsigned int* dProcCounts = idx_.l0Bufs.dProcCounts.data();
    size_t* dCompactIds = idx_.l0Bufs.dCompactIds.data();
    double* dCompactVals = idx_.l0Bufs.dCompactVals.data();
    double* dYValue = idx_.l0Bufs.dYValue[l].data();
    size_t* hVecIdx = idx_.l0Bufs.hVectorIndex;
    size_t* hProcCol = idx_.l0Bufs.hProcColInd;
    size_t* hProcRow = idx_.l0Bufs.hProcRowInd;
    size_t* hProcMatPos = idx_.l0Bufs.hProcMatrixPos;

    size_t selectCount = 0;

    if (qType == QueryType::RANGE) {
        CUDA_CHECK(cudaMemcpy(dLo, lo, D * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dHi, hi, D * sizeof(double), cudaMemcpyHostToDevice));

        const dim3 block(256);
        const dim3 grid((p + 255) / 256);
        rangePruningKernel<<<grid, block>>>(dMask, dLo, dHi, currentGrid->get_vals_(), dRadius, D, p,
                                            currentGrid->get_ids_());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Count selected (host-side, download mask)
        // Reuse hVecIdx as temporary bool storage (byte-compatible, capacity maxP)
        auto* hMask = reinterpret_cast<bool*>(hVecIdx);
        CUDA_CHECK(cudaMemcpy(hMask, dMask, p * sizeof(bool), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < p; ++i) {
            if (hMask[i]) {
                ++selectCount;
            }
        }
    }
    else {
        CUDA_CHECK(cudaMemcpy(dQueryPoint, queryPoint, D * sizeof(double), cudaMemcpyHostToDevice));

        // Distance kernel
        const dim3 block(256);
        const dim3 grid_k((p + 255) / 256);
        calculateClusterDistanceKernel<<<grid_k, block>>>(dClusters, dQueryPoint, currentGrid->get_vals_(), dRadius, D,
                                                          currentGrid->get_ids_(), p);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Sort on device (in-place)
        constexpr int kIPB = 512;
        constexpr int kIPT = 2;
        Cluster sentinel{};
        sentinel.distance = std::numeric_limits<double>::max();
        sentinel.label = 0;
        gpuSort<kIPB, kIPT, true>(static_cast<Cluster*>(dClusters), static_cast<Cluster*>(dClusters),
                                  static_cast<int>(p), sentinel);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Download sorted clusters and ids into pre-allocated host buffers
        auto* hCl = reinterpret_cast<Cluster*>(idx_.l0Bufs.hClusters);
        CUDA_CHECK(cudaMemcpy(hCl, dClusters, p * sizeof(Cluster), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hVecIdx, currentGrid->get_ids_(), p * sizeof(size_t), cudaMemcpyDeviceToHost));

        // l0Bufs.hClusters allocation: maxNnz*sizeof(Cluster) + maxNnz bytes
        // Use the trailing maxNnz bytes as bool mask scratch (p <= maxNnz)
        auto* hMask = reinterpret_cast<bool*>(idx_.l0Bufs.hClusters + p * sizeof(Cluster));
        std::fill(hMask, hMask + p, false);

        const double* hRadius = idx_.meshMaxRadius[l];
        const size_t* hClusterSizes = idx_.pTensors[l]->get_nnz_per_col();
        size_t currentCount = 0;
        double currentDistance = 0.0;

        for (size_t i = 0; i < p; ++i) {
            const size_t globalIdx = hVecIdx[static_cast<size_t>(hCl[i].label)];
            if (currentCount >= K) {
                if (hCl[i].distance > currentDistance) {
                    break;
                }
                currentCount += hClusterSizes[globalIdx];
                hMask[static_cast<size_t>(hCl[i].label)] = true;
                ++selectCount;
            }
            else {
                currentCount += hClusterSizes[globalIdx];
                currentDistance = std::max(currentDistance, hCl[i].distance + 2.0 * hRadius[globalIdx]);
                hMask[static_cast<size_t>(hCl[i].label)] = true;
                ++selectCount;
            }
        }

        // Upload mask
        CUDA_CHECK(cudaMemcpy(dMask, hMask, p * sizeof(bool), cudaMemcpyHostToDevice));
    }

    // Compact  (direct kernel calls, pre-allocated buffers)
    const size_t nProc = (p + kCompactKp - 1) / kCompactKp;

    launchCountKernel<bool, kCompactKp, kCompactBlockSize>(dProcCounts, dMask, static_cast<unsigned int>(p));
    CUDA_CHECK(cudaDeviceSynchronize());

    launchPrefixSumKernel<unsigned int, kCompactBlockSize>(dProcCounts, static_cast<unsigned int>(nProc));

    const size_t gridSz = (nProc + kCompactWPB - 1) / kCompactWPB;
    gridCompactPrefixKernel<kCompactIter>
        <<<gridSz, kCompactBlockSize>>>(dCompactVals, dCompactIds, currentGrid->get_vals_(), currentGrid->get_ids_(),
                                        dMask, dProcCounts, static_cast<unsigned int>(p), D, nProc, selectCount);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Non-owning view over compact output
    SparseGrid pruned{currentGrid->get_num_rows(), D, selectCount, dCompactIds, dCompactVals};

    // SpTSpM  (no-alloc variant)
    SparseGrid* fineGrid = SpTSpMMultiplication_v3_L0_nb(
        idx_.pTensors[l], &pruned, const_cast<double*>(dPVals), idx_.dSpTSpMBufs[l].colInd.data(),
        idx_.dSpTSpMBufs[l].rowInd.data(), idx_.dSpTSpMBufs[l].matrixPos.data(), dYValue, hVecIdx, hProcCol, hProcRow,
        hProcMatPos);

    // Refactor
    refactor(*fineGrid, dMap);

    std::cout << "  Level " << l << " [L0] -> " << fineGrid->get_nnz_nums() << " pts\n";

    // fineGrid->ids_  = dSpTSpMBufs[l].rowInd (IndexData-owned)
    // fineGrid->vals_ = l0Bufs.dYValue[l]      (IndexData-owned)
    return {fineGrid, /*ownsIds=*/false, /*ownsVals=*/false};
}
