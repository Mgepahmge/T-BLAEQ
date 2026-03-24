#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include "IndexData.cuh"

static std::string fmtBytes(size_t b) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(2);
    if (b >= (1ULL << 30)) {
        o << b / double(1ULL << 30) << " GiB";
    }
    else if (b >= (1ULL << 20)) {
        o << b / double(1ULL << 20) << " MiB";
    }
    else if (b >= (1ULL << 10)) {
        o << b / double(1ULL << 10) << " KiB";
    }
    else {
        o << b << " B";
    }
    return o.str();
}

void MemoryStats::print(std::ostream& os) const {
    os << "\nIndex Memory Usage\n";
    os << "[HOST]\n";
    os << "    P-tensor arrays : " << fmtBytes(hostPTensors) << "\n";
    os << "    Sort maps : " << fmtBytes(hostMaps) << "\n";
    os << "    Cluster radius : " << fmtBytes(hostRadius) << "\n";
    os << "    Coarsest mesh : " << fmtBytes(hostCoarsestMesh) << "\n";
    os << "    HOST TOTAL : " << fmtBytes(hostTotal) << "\n";
    os << "[DEVICE]\n";
    os << "    P-tensor vals : " << fmtBytes(devicePTensorVals) << "\n";
    os << "    Sort maps : " << fmtBytes(deviceMaps) << "\n";
    os << "    Cluster radius : " << fmtBytes(deviceRadius) << "\n";
    os << "    Coarsest mesh : " << fmtBytes(deviceCoarsestMesh) << "\n";
    os << "    SpTSpM bufs (L0) : " << fmtBytes(deviceSpTSpMBufs) << "  (additional if any level uses L0)\n";
    os << "    DEVICE TOTAL : " << fmtBytes(deviceTotal) << "\n";
    os << "GRAND TOTAL : " << fmtBytes(grandTotal) << "\n";
    os << "\n\n";
}

std::string MemoryStats::toCsv() const {
    std::ostringstream o;
    o << hostTotal << ',' << deviceTotal << ',' << grandTotal;
    return o.str();
}

void IndexData::uploadPermanentData() {
    if (permanentDataOnDevice) {
        return;
    }

    // Maps
    dMaps.resize(height);
    for (size_t i = 1; i < height; ++i) {
        if (maps[i] == nullptr) {
            continue;
        }
        const size_t len = meshSizes[intervals - i];
        dMaps[i].reset(maps[i], len);
    }

    // Coarsest mesh
    const size_t nnz = coarsestMesh->get_nnz_nums();
    const size_t dim = coarsestMesh->get_dimensions();
    dCoarsestMeshIds.reset(coarsestMesh->get_ids_(), nnz);
    dCoarsestMeshVals.reset(coarsestMesh->get_vals_(), nnz * dim);

    permanentDataOnDevice = true;
}

void IndexData::allocSpTSpMBuffers() {
    if (spTSpMBufsReady) {
        return;
    }

    dSpTSpMBufs.resize(intervals);
    for (size_t l = 0; l < intervals; ++l) {
        if (activePolicy.levels[l] != LevelPolicy::L0) {
            continue;
        }
        const size_t cap = pTensors[l]->get_row_nums();
        dSpTSpMBufs[l].colInd.resize(cap);
        dSpTSpMBufs[l].rowInd.resize(cap);
        dSpTSpMBufs[l].matrixPos.resize(cap);
    }
    spTSpMBufsReady = true;
}

void IndexData::allocL0WorkBuffers() {
    if (l0Bufs.ready) {
        return;
    }

    size_t maxNnz = 0;
    for (size_t l = 0; l < intervals; ++l) {
        maxNnz = std::max(maxNnz, pTensors[l]->get_nnz_nums());
    }
    const size_t maxNProc = (maxNnz + 127) / 128;

    l0Bufs.dMask.resize(maxNnz);
    l0Bufs.dClusters.resize(maxNnz * 16);
    l0Bufs.dQueryPoint.resize(D);
    l0Bufs.dLo.resize(D);
    l0Bufs.dHi.resize(D);
    l0Bufs.dProcCounts.resize(maxNProc);
    l0Bufs.dCompactIds.resize(maxNnz);
    l0Bufs.dCompactVals.resize(maxNnz * D);

    l0Bufs.dYValue.resize(intervals);
    for (size_t l = 0; l < intervals; ++l) {
        l0Bufs.dYValue[l].resize(pTensors[l]->get_nnz_nums() * D);
    }

    l0Bufs.hVectorIndex = new size_t[maxNnz];
    l0Bufs.hClusters = new uint8_t[maxNnz * 16 + maxNnz];
    l0Bufs.hProcColInd = new size_t[maxNnz];
    l0Bufs.hProcRowInd = new size_t[maxNnz];
    l0Bufs.hProcMatrixPos = new size_t[maxNnz];

    l0Bufs.ready = true;
}

void IndexData::computeStats() {
    stats = MemoryStats{};

    // Host
    for (const auto* t : pTensors) {
        const size_t nnz = t->get_nnz_nums();
        const size_t col = t->get_col_nums();
        const size_t d = t->get_dim();
        stats.hostPTensors += nnz * d * sizeof(double) + col * sizeof(size_t) // nnzPerCol
            + (col + 1) * sizeof(size_t) // colRes
            + nnz * sizeof(size_t); // rowIds
    }
    for (size_t i = 1; i < height; ++i) {
        if (maps[i]) {
            stats.hostMaps += meshSizes[intervals - i] * sizeof(size_t);
        }
    }
    for (size_t i = 0; i < intervals; ++i) {
        stats.hostRadius += meshSizes[intervals - i] * sizeof(double);
    }
    if (coarsestMesh) {
        const size_t nnz = coarsestMesh->get_nnz_nums();
        const size_t dim = coarsestMesh->get_dimensions();
        stats.hostCoarsestMesh = nnz * sizeof(size_t) + nnz * dim * sizeof(double);
    }
    stats.hostTotal = stats.hostPTensors + stats.hostMaps + stats.hostRadius + stats.hostCoarsestMesh;

    // Device (L1 baseline: treat all levels as L1 for reference)
    for (const auto* t : pTensors) {
        stats.devicePTensorVals += t->get_nnz_nums() * t->get_dim() * sizeof(double);
    }
    for (size_t i = 1; i < height; ++i) {
        if (!dMaps[i].empty()) {
            stats.deviceMaps += dMaps[i].size() * sizeof(size_t);
        }
    }
    for (size_t i = 0; i < intervals; ++i) {
        stats.deviceRadius += meshSizes[intervals - i] * sizeof(double);
    }
    stats.deviceCoarsestMesh = dCoarsestMeshIds.size() * sizeof(size_t) + dCoarsestMeshVals.size() * sizeof(double);

    // L0 SpTSpM pre-alloc (if all levels L0)
    for (size_t l = 0; l < intervals; ++l) {
        stats.deviceSpTSpMBufs += 3 * pTensors[l]->get_row_nums() * sizeof(size_t);
    }

    stats.deviceTotal = stats.devicePTensorVals + stats.deviceMaps + stats.deviceRadius + stats.deviceCoarsestMesh;

    stats.grandTotal = stats.hostTotal + stats.deviceTotal;
}

void IndexData::release() {
    releasePTensors();
    releaseMaps();
    releaseRadius();

    dPTensorVals.clear();
    dMeshMaxRadius.clear();
    dMaps.clear();
    dSpTSpMBufs.clear();
    dCoarsestMeshIds.free();
    dCoarsestMeshVals.free();

    l0Bufs.release();

    if (coarsestMesh) {
        delete[] coarsestMesh->get_ids_();
        delete[] coarsestMesh->get_vals_();
        coarsestMesh->set_ids(nullptr);
        coarsestMesh->set_vals(nullptr);
        delete coarsestMesh;
        coarsestMesh = nullptr;
    }
}

void IndexData::releasePTensors() {
    for (auto* t : pTensors) {
        delete t;
    }
    pTensors.clear();
}

void IndexData::releaseMaps() {
    for (auto* m : maps) {
        delete[] m;
    }
    maps.clear();
}

void IndexData::releaseRadius() {
    for (auto* r : meshMaxRadius) {
        delete[] r;
    }
    meshMaxRadius.clear();
}
