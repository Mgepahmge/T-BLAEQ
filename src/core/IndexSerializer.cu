#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include "IndexSerializer.cuh"
#include "src/func.hpp"

namespace fs = std::filesystem;

void IndexSerializer::save(const IndexData& d, const std::string& dirPath) {
    fs::create_directories(dirPath);

    saveMetadata(d, dirPath + "/metadata.bin");

    for (size_t i = 0; i < d.intervals; ++i) {
        saveSparseTensorCsc(d.pTensors[i], dirPath + "/P_Tensor_" + std::to_string(i) + ".bin");
    }

    for (size_t i = 0; i < d.intervals; ++i) {
        const size_t count = d.meshSizes[d.intervals - i];
        saveMaxRadius(d.meshMaxRadius[i], count, dirPath + "/MaxRadius_" + std::to_string(i) + ".bin");
    }

    for (size_t i = 1; i < d.height; ++i) {
        if (d.maps[i]) {
            saveMap(d.maps[i], d.meshSizes[d.intervals - i], dirPath + "/Map_" + std::to_string(i) + ".bin");
        }
    }

    saveCoarsestMesh(d.coarsestMesh, dirPath + "/Coreast_Mesh.bin");
}

IndexData* IndexSerializer::load(const std::string& dirPath) {
    auto* d = new IndexData();

    loadMetadata(*d, dirPath + "/metadata.bin");

    d->pTensors.resize(d->intervals, nullptr);
    d->meshMaxRadius.resize(d->intervals, nullptr);
    d->maps.resize(d->height, nullptr);
    d->dMaps.resize(d->height); // DeviceBuffer is default-constructible (empty)

    for (size_t i = 0; i < d->intervals; ++i) {
        d->pTensors[i] = loadSparseTensorCsc(dirPath + "/P_Tensor_" + std::to_string(i) + ".bin");
    }

    for (size_t i = 0; i < d->intervals; ++i) {
        size_t count = 0;
        d->meshMaxRadius[i] = loadMaxRadius(dirPath + "/MaxRadius_" + std::to_string(i) + ".bin", count);
        assert(count == d->meshSizes[d->intervals - i]);
    }

    for (size_t i = 1; i < d->height; ++i) {
        const std::string path = dirPath + "/Map_" + std::to_string(i) + ".bin";
        if (fs::exists(path)) {
            size_t sz = 0;
            d->maps[i] = loadMap(path, sz);
            assert(sz == d->meshSizes[d->intervals - i]);
        }
    }

    d->coarsestMesh = loadCoarsestMesh(dirPath + "/Coreast_Mesh.bin");

    std::cout << "Index loaded successfully from: " << dirPath << "\n";
    return d;
}

void IndexSerializer::saveMetadata(const IndexData& d, const std::string& path) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Cannot write: " + path);
    }

    ofs.write(reinterpret_cast<const char*>(&d.D), sizeof(d.D));
    ofs.write(reinterpret_cast<const char*>(&d.N), sizeof(d.N));
    ofs.write(reinterpret_cast<const char*>(&d.height), sizeof(d.height));
    ofs.write(reinterpret_cast<const char*>(&d.intervals), sizeof(d.intervals));
    ofs.write(reinterpret_cast<const char*>(&d.isAosArch), sizeof(d.isAosArch));

    for (size_t i = 0; i < d.intervals; ++i) {
        ofs.write(reinterpret_cast<const char*>(&d.ratios[i]), sizeof(size_t));
    }

    const size_t msz = d.meshSizes.size();
    ofs.write(reinterpret_cast<const char*>(&msz), sizeof(msz));
    ofs.write(reinterpret_cast<const char*>(d.meshSizes.data()), msz * sizeof(size_t));
}

void IndexSerializer::loadMetadata(IndexData& d, const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Cannot read: " + path);
    }

    ifs.read(reinterpret_cast<char*>(&d.D), sizeof(d.D));
    ifs.read(reinterpret_cast<char*>(&d.N), sizeof(d.N));
    ifs.read(reinterpret_cast<char*>(&d.height), sizeof(d.height));
    ifs.read(reinterpret_cast<char*>(&d.intervals), sizeof(d.intervals));
    ifs.read(reinterpret_cast<char*>(&d.isAosArch), sizeof(d.isAosArch));

    d.ratios.resize(d.intervals);
    for (size_t i = 0; i < d.intervals; ++i) {
        ifs.read(reinterpret_cast<char*>(&d.ratios[i]), sizeof(size_t));
    }

    size_t msz = 0;
    ifs.read(reinterpret_cast<char*>(&msz), sizeof(msz));
    d.meshSizes.resize(msz);
    ifs.read(reinterpret_cast<char*>(d.meshSizes.data()), msz * sizeof(size_t));

    std::cout << "Metadata loaded: D=" << d.D << " N=" << d.N << " height=" << d.height << " intervals=" << d.intervals
              << "\n";
}

void IndexSerializer::saveSparseTensorCsc(const SparseTensorCscFormat* t, const std::string& path) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Cannot write: " + path);
    }

    const size_t D = t->get_dim();
    const size_t rowNums = t->get_row_nums();
    const size_t colNums = t->get_col_nums();
    const size_t nnzNums = t->get_nnz_nums();
    const bool isAos = t->get_memory_arch();

    ofs.write(reinterpret_cast<const char*>(&D), sizeof(D));
    ofs.write(reinterpret_cast<const char*>(&rowNums), sizeof(rowNums));
    ofs.write(reinterpret_cast<const char*>(&colNums), sizeof(colNums));
    ofs.write(reinterpret_cast<const char*>(&nnzNums), sizeof(nnzNums));
    ofs.write(reinterpret_cast<const char*>(&isAos), sizeof(isAos));

    ofs.write(reinterpret_cast<const char*>(t->get_row_ids()), nnzNums * sizeof(size_t));
    ofs.write(reinterpret_cast<const char*>(t->get_nnz_per_col()), colNums * sizeof(size_t));
    ofs.write(reinterpret_cast<const char*>(t->get_col_res()), (colNums + 1) * sizeof(size_t));
    ofs.write(reinterpret_cast<const char*>(t->get_vals()), nnzNums * D * sizeof(double));
}

SparseTensorCscFormat* IndexSerializer::loadSparseTensorCsc(const std::string& path) {

    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Cannot read: " + path);
    }

    size_t D, rowNums, colNums, nnzNums;
    bool isAos;

    ifs.read(reinterpret_cast<char*>(&D), sizeof(D));
    ifs.read(reinterpret_cast<char*>(&rowNums), sizeof(rowNums));
    ifs.read(reinterpret_cast<char*>(&colNums), sizeof(colNums));
    ifs.read(reinterpret_cast<char*>(&nnzNums), sizeof(nnzNums));
    ifs.read(reinterpret_cast<char*>(&isAos), sizeof(isAos));

    // Read nnz_per_col first to use the constructor that pre-builds col_res.
    // Layout on disk: row_ids | nnz_per_col | col_res | vals
    // We need nnz_per_col to construct; then patch row_ids, col_res, vals.
    std::vector<size_t> nnzPerCol(colNums);
    // Skip row_ids for now, read nnz_per_col
    const std::streamoff rowIdsBytes = static_cast<std::streamoff>(nnzNums * sizeof(size_t));
    ifs.seekg(rowIdsBytes, std::ios::cur);
    ifs.read(reinterpret_cast<char*>(nnzPerCol.data()), colNums * sizeof(size_t));
    // Rewind to row_ids
    ifs.seekg(-(rowIdsBytes + static_cast<std::streamoff>(colNums * sizeof(size_t))), std::ios::cur);

    auto* t = new SparseTensorCscFormat(D, rowNums, colNums, nnzPerCol);

    ifs.read(reinterpret_cast<char*>(t->getRowIdsMut()), nnzNums * sizeof(size_t));
    // Skip nnz_per_col (already consumed by constructor)
    ifs.seekg(static_cast<std::streamoff>(colNums * sizeof(size_t)), std::ios::cur);
    ifs.read(reinterpret_cast<char*>(t->getColResMut()), (colNums + 1) * sizeof(size_t));
    ifs.read(reinterpret_cast<char*>(t->getValsMut()), nnzNums * D * sizeof(double));

    return t;
}

void IndexSerializer::saveMaxRadius(const double* radius, size_t count, const std::string& path) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Cannot write: " + path);
    }
    ofs.write(reinterpret_cast<const char*>(&count), sizeof(count));
    ofs.write(reinterpret_cast<const char*>(radius), count * sizeof(double));
}

double* IndexSerializer::loadMaxRadius(const std::string& path, size_t& countOut) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Cannot read: " + path);
    }
    ifs.read(reinterpret_cast<char*>(&countOut), sizeof(countOut));
    auto* r = new double[countOut];
    ifs.read(reinterpret_cast<char*>(r), countOut * sizeof(double));
    return r;
}

void IndexSerializer::saveMap(const size_t* map, size_t size, const std::string& path) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Cannot write: " + path);
    }
    ofs.write(reinterpret_cast<const char*>(&size), sizeof(size));
    ofs.write(reinterpret_cast<const char*>(map), size * sizeof(size_t));
}

size_t* IndexSerializer::loadMap(const std::string& path, size_t& sizeOut) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Cannot read: " + path);
    }
    ifs.read(reinterpret_cast<char*>(&sizeOut), sizeof(sizeOut));
    auto* m = new size_t[sizeOut];
    ifs.read(reinterpret_cast<char*>(m), sizeOut * sizeof(size_t));
    return m;
}

void IndexSerializer::saveCoarsestMesh(const GridAsSparseMatrix* mesh, const std::string& path) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Cannot write: " + path);
    }

    const size_t mRow = mesh->get_num_rows();
    const size_t mDim = mesh->get_dimensions();
    const size_t mNnz = mesh->get_nnz_nums();
    const bool isAos = mesh->get_memory_arch();

    ofs.write(reinterpret_cast<const char*>(&mRow), sizeof(mRow));
    ofs.write(reinterpret_cast<const char*>(&mDim), sizeof(mDim));
    ofs.write(reinterpret_cast<const char*>(&mNnz), sizeof(mNnz));
    ofs.write(reinterpret_cast<const char*>(&isAos), sizeof(isAos));
    ofs.write(reinterpret_cast<const char*>(mesh->get_ids_()), mNnz * sizeof(size_t));
    ofs.write(reinterpret_cast<const char*>(mesh->get_vals_()), mNnz * mDim * sizeof(double));
}

GridAsSparseMatrix* IndexSerializer::loadCoarsestMesh(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Cannot read: " + path);
    }

    size_t mRow, mDim, mNnz;
    bool isAos;

    ifs.read(reinterpret_cast<char*>(&mRow), sizeof(mRow));
    ifs.read(reinterpret_cast<char*>(&mDim), sizeof(mDim));
    ifs.read(reinterpret_cast<char*>(&mNnz), sizeof(mNnz));
    ifs.read(reinterpret_cast<char*>(&isAos), sizeof(isAos));

    auto* ids = new size_t[mNnz];
    auto* vals = new double[mNnz * mDim];
    ifs.read(reinterpret_cast<char*>(ids), mNnz * sizeof(size_t));
    ifs.read(reinterpret_cast<char*>(vals), mNnz * mDim * sizeof(double));

    auto* mesh = new GridAsSparseMatrix(mRow, mDim, mNnz, ids, vals);
    mesh->set_memory_arch(isAos);
    return mesh;
}
