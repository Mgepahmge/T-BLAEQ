#include <cuda_runtime.h>
#include "StrategyCommon.cuh"
#include "src/Query/KNN.cuh"
#include "src/Query/RangePruning.cuh"
#include "src/core/Memory.cuh"

void freeDeviceSparseGrid(SparseGrid* g, bool ownsIds, bool ownsVals) {
    if (!g) {
        return;
    }
    if (ownsIds && g->get_ids_()) {
        cudaFree(g->get_ids_());
    }
    if (ownsVals && g->get_vals_()) {
        cudaFree(g->get_vals_());
    }
    g->set_ids(nullptr);
    g->set_vals(nullptr);
    delete g;
}

bool* runPruning(size_t l, IndexData& idx, SparseGrid* currentGrid, QueryType qType, const double* lo, const double* hi,
                 const double* queryPoint, size_t K, size_t& selectCount) {
    const double* dRadius = idx.dMeshMaxRadius[l].data();

    if (qType == QueryType::RANGE) {
        DeviceBuffer<double> dLo(lo, idx.D);
        DeviceBuffer<double> dHi(hi, idx.D);
        return rangePruning(dLo.data(), dHi.data(), idx.D, currentGrid->get_vals_(), dRadius,
                            currentGrid->get_nnz_nums(), currentGrid->get_ids_(), currentGrid->get_num_rows(),
                            selectCount);
    }
    else {
        const double* hRadius = idx.meshMaxRadius[l];
        const size_t* hClusterSizes = idx.pTensors[l]->get_nnz_per_col();
        return knnPruning(K, currentGrid->get_nnz_nums(), idx.D, currentGrid->get_num_rows(), queryPoint,
                          currentGrid->get_vals_(), dRadius, hRadius, hClusterSizes, currentGrid->get_ids_(),
                          selectCount);
    }
}
