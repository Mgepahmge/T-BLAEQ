#include <cassert>
#include <iostream>
#include <limits>
#include "L2Strategy.cuh"
#include "StrategyCommon.cuh"
#include "src/Kernel/SpMSpV.cuh"
#include "src/Query/GridCompact.cuh"
#include "src/Query/KNN.cuh"
#include "src/Query/RangePruning.cuh"
#include "src/Query/Refactor.cuh"
#include "src/Query/compact.cuh"
#include "src/core/Memory.cuh"
#include "src/core/MemoryPolicy.cuh"
#include "src/utils/NVTXProfiler.cuh"


void L2Strategy::prepare(std::ostream& reportOs) {
    idx_.computeStats();
    PolicyScheduler::printReport(idx_, idx_.activePolicy, reportOs);
    idx_.uploadPermanentData();
    // Resize vectors so per-level upload/free calls can index into them.
    // Buffers stay empty (nullptr) until needed.
    idx_.dPTensorVals.resize(idx_.intervals);
    idx_.dMeshMaxRadius.resize(idx_.intervals);
}

LevelResult L2Strategy::runLevel(size_t l, SparseGrid* currentGrid, QueryType qType, const double* lo, const double* hi,
                                 const double* queryPoint, size_t K) {
    assert(idx_.permanentDataOnDevice);

    NvtxProfiler lp(("L2_Level" + std::to_string(l)).c_str(), NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Green);

    const size_t D = idx_.D;
    const size_t* dMap = idx_.dMaps[l + 1].data();

    // Step 1: Pruning
    // Upload radius, run kernel, download mask, FREE radius immediately.
    {
        const size_t rLen = idx_.meshSizes[idx_.intervals - l];
        idx_.dMeshMaxRadius[l].reset(idx_.meshMaxRadius[l], rLen);
    }
    const double* dRadius = idx_.dMeshMaxRadius[l].data();

    size_t selectCount = 0;
    bool* mask = nullptr;

    if (qType == QueryType::RANGE) {
        DeviceBuffer<double> dLo(lo, D);
        DeviceBuffer<double> dHi(hi, D);
        mask = rangePruning(dLo.data(), dHi.data(), D, currentGrid->get_vals_(), dRadius, currentGrid->get_nnz_nums(),
                            currentGrid->get_ids_(), currentGrid->get_num_rows(), selectCount);
    }
    else {
        const double* hRadius = idx_.meshMaxRadius[l];
        const size_t* hClusterSizes = idx_.pTensors[l]->get_nnz_per_col();
        mask =
            knnPruning(K, currentGrid->get_nnz_nums(), D, currentGrid->get_num_rows(), queryPoint,
                       currentGrid->get_vals_(), dRadius, hRadius, hClusterSizes, currentGrid->get_ids_(), selectCount);
    }

    // Radius no longer needed -- free immediately
    idx_.dMeshMaxRadius[l].free();

    // Step 2: Compact
    // currentGrid is device-resident; compact produces a new device grid.
    SparseGrid* pruned = compactGrid(*currentGrid, mask, selectCount);
    delete[] mask;

    // Step 3: SpTSpM
    // L2 variant: only upload the columns corresponding to pruned centroids.
    // P vals stay on host; v3_L2 selectively uploads only what is needed.
    SparseGrid* fineGrid = SpTSpMMultiplication_v3_L2(idx_.pTensors[l], pruned);

    // Step 4: Refactor using permanent dMaps
    refactor(*fineGrid, dMap);

    // Cleanup compact output
    freeDeviceSparseGrid(pruned, /*ownsIds=*/true);

    std::cout << "  Level " << l << " [L2] -> " << fineGrid->get_nnz_nums() << " pts\n";

    return {fineGrid, /*ownsIds=*/true};
}
