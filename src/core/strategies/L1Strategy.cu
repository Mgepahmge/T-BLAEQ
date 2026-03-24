#include <cassert>
#include <iostream>
#include "L1Strategy.cuh"
#include "StrategyCommon.cuh"
#include "src/Kernel/SpMSpV.cuh"
#include "src/Query/GridCompact.cuh"
#include "src/Query/Refactor.cuh"
#include "src/core/MemoryPolicy.cuh"
#include "src/utils/NVTXProfiler.cuh"


void L1Strategy::prepare(std::ostream& reportOs) {
    idx_.computeStats();
    PolicyScheduler::printReport(idx_, idx_.activePolicy, reportOs);
    idx_.uploadPermanentData();
    idx_.dPTensorVals.resize(idx_.intervals);
    idx_.dMeshMaxRadius.resize(idx_.intervals);
    for (size_t l = 0; l < idx_.intervals; ++l) {
        const size_t vLen = idx_.pTensors[l]->get_nnz_nums() * idx_.pTensors[l]->get_dim();
        idx_.dPTensorVals[l].reset(idx_.pTensors[l]->get_vals(), vLen);
        const size_t rLen = idx_.meshSizes[idx_.intervals - l];
        idx_.dMeshMaxRadius[l].reset(idx_.meshMaxRadius[l], rLen);
    }
}

LevelResult L1Strategy::runLevel(size_t l, SparseGrid* currentGrid, QueryType qType, const double* lo, const double* hi,
                                 const double* queryPoint, size_t K) {
    assert(idx_.permanentDataOnDevice);


    NvtxProfiler lp(("L1_Level" + std::to_string(l)).c_str(), NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Green);

    const double* dRadius = idx_.dMeshMaxRadius[l].data();
    const double* dTensorVal = idx_.dPTensorVals[l].data();
    const size_t* dMap = idx_.dMaps[l + 1].data();
    assert(dRadius != nullptr && dTensorVal != nullptr);

    // Pruning
    size_t selectCount = 0;
    bool* mask = runPruning(l, idx_, currentGrid, qType, lo, hi, queryPoint, K, selectCount);

    // Compact
    SparseGrid* pruned = compactGrid(*currentGrid, mask, selectCount);
    delete[] mask;

    // SpTSpM (L1: dynamic index buffer allocation inside v3)
    SparseGrid* fineGrid = SpTSpMMultiplication_v3(idx_.pTensors[l], pruned, const_cast<double*>(dTensorVal));

    // refactor
    refactor(*fineGrid, dMap);

    // Cleanup
    freeDeviceSparseGrid(pruned, /*ownsIds=*/true);

    std::cout << "  Level " << l << " [L1] -> " << fineGrid->get_nnz_nums() << " pts\n";

    return {fineGrid, /*ownsIds=*/true};
}
