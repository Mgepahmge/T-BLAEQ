#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include "Data_Structures.cuh"
#include "src/func.hpp"

SparseGrid::SparseGrid(const std::vector<PointVec>& points, const std::vector<size_t>& rowIds, size_t logicRows,
                       size_t /*dim*/) {
    if (points.empty()) {
        numRows_ = logicRows;
        return;
    }

    assert(logicRows >= points.size());
    assert(points.size() == rowIds.size());

    numRows_ = logicRows;
    dim_ = points[0].size();
    nnz_ = rowIds.size();
    ids_ = new size_t[nnz_];
    vals_ = new double[nnz_ * dim_]();

    std::copy(rowIds.begin(), rowIds.end(), ids_);
    for (size_t i = 0; i < points.size(); ++i) {
        assert(points[i].size() == dim_);
        for (size_t j = 0; j < dim_; ++j) {
            vals_[i * dim_ + j] = points[i][j];
        }
    }
}

SparseGrid::SparseGrid(const std::vector<PointVec>& points, size_t begin, size_t last) {
    if (points.empty()) {
        return;
    }
    assert(begin <= last && last <= points.size());

    numRows_ = points.size();
    dim_ = points[0].size();
    nnz_ = last - begin;
    ids_ = new size_t[nnz_]();
    vals_ = new double[nnz_ * dim_]();

    size_t w = 0;
    for (size_t i = begin; i < last; ++i) {
        ids_[w] = i;
        for (size_t j = 0; j < dim_; ++j) {
            vals_[w * dim_ + j] = points[i][j];
        }
        ++w;
    }
}

SparseGrid::SparseGrid(const SparseGrid& o) :
    isAos_(o.isAos_), numRows_(o.numRows_), dim_(o.dim_), nnz_(o.nnz_), ownsBuffers_(true) {
    if (!o.ownsBuffers_) {
        throw std::logic_error("SparseGrid: cannot copy a device-memory wrapper (ownsBuffers=false).");
    }

    if (o.ids_) {
        ids_ = new size_t[nnz_];
        std::copy(o.ids_, o.ids_ + nnz_, ids_);
    }
    if (o.vals_) {
        vals_ = new double[nnz_ * dim_];
        std::copy(o.vals_, o.vals_ + nnz_ * dim_, vals_);
    }
}

SparseGrid& SparseGrid::operator=(const SparseGrid& o) {
    if (this == &o) {
        return *this;
    }
    if (!o.ownsBuffers_) {
        throw std::logic_error("SparseGrid: cannot copy-assign from a device-memory wrapper.");
    }

    if (ownsBuffers_) {
        delete[] ids_;
        delete[] vals_;
    }

    isAos_ = o.isAos_;
    numRows_ = o.numRows_;
    dim_ = o.dim_;
    nnz_ = o.nnz_;
    ownsBuffers_ = true;
    ids_ = nullptr;
    vals_ = nullptr;

    if (o.ids_) {
        ids_ = new size_t[nnz_];
        std::copy(o.ids_, o.ids_ + nnz_, ids_);
    }
    if (o.vals_) {
        vals_ = new double[nnz_ * dim_];
        std::copy(o.vals_, o.vals_ + nnz_ * dim_, vals_);
    }
    return *this;
}

SparseGrid::SparseGrid(SparseGrid&& o) noexcept :
    isAos_(o.isAos_), numRows_(o.numRows_), dim_(o.dim_), nnz_(o.nnz_), ids_(o.ids_), vals_(o.vals_),
    ownsBuffers_(o.ownsBuffers_) {
    o.ids_ = nullptr;
    o.vals_ = nullptr;
    o.numRows_ = 0;
    o.dim_ = 0;
    o.nnz_ = 0;
    o.ownsBuffers_ = true;
}

SparseGrid& SparseGrid::operator=(SparseGrid&& o) noexcept {
    if (this == &o) {
        return *this;
    }
    if (ownsBuffers_) {
        delete[] ids_;
        delete[] vals_;
    }

    isAos_ = o.isAos_;
    numRows_ = o.numRows_;
    dim_ = o.dim_;
    nnz_ = o.nnz_;
    ids_ = o.ids_;
    vals_ = o.vals_;
    ownsBuffers_ = o.ownsBuffers_;

    o.ids_ = nullptr;
    o.vals_ = nullptr;
    o.numRows_ = 0;
    o.dim_ = 0;
    o.nnz_ = 0;
    o.ownsBuffers_ = true;
    return *this;
}

SparseGrid::~SparseGrid() {
    if (ownsBuffers_) {
        delete[] ids_;
        delete[] vals_;
    }
}

void SparseGrid::setIdsFromVec(std::vector<size_t>& v) {
    assert(ids_ == nullptr);
    nnz_ = v.size();
    ids_ = new size_t[nnz_];
    std::copy(v.begin(), v.end(), ids_);
}

void SparseGrid::freeDram() {
    assert(ownsBuffers_);
    delete[] ids_;
    ids_ = nullptr;
    delete[] vals_;
    vals_ = nullptr;
}

SparseTensorCooFormat::SparseTensorCooFormat(size_t D, size_t rowNums, size_t colNums) :
    D_(D), rowNums_(rowNums), colNums_(colNums), nnzNums_(rowNums) {
    rowIds_ = new size_t[nnzNums_]();
    colIds_ = new size_t[nnzNums_]();
    vals_ = new double[nnzNums_ * D_]();
}

SparseTensorCooFormat::~SparseTensorCooFormat() {
    delete[] rowIds_;
    delete[] colIds_;
    delete[] vals_;
}

void SparseTensorCooFormat::insertNnz(size_t row, size_t col, double* valVec) {
    rowIds_[writeIdx_] = row;
    colIds_[writeIdx_] = col;
    for (size_t i = 0; i < D_; ++i) {
        vals_[writeIdx_ * D_ + i] = valVec[i];
    }
    ++writeIdx_;
}

void SparseTensorCooFormat::display(const std::string& filename) const {
    std::ofstream of(filename);
    if (!of) {
        throw std::runtime_error("Cannot open: " + filename);
    }
    of << std::fixed << std::setprecision(3);

    std::vector<std::vector<int>> map(rowNums_, std::vector<int>(colNums_, -1));
    for (size_t i = 0; i < nnzNums_; ++i) {
        map[rowIds_[i]][colIds_[i]] = static_cast<int>(i);
    }

    for (size_t i = 0; i < rowNums_; ++i) {
        of << '{';
        for (size_t j = 0; j < colNums_; ++j) {
            of << '[';
            if (map[i][j] >= 0) {
                size_t idx = static_cast<size_t>(map[i][j]);
                for (size_t k = 0; k < D_; ++k) {
                    if (k) {
                        of << ',';
                    }
                    of << vals_[idx * D_ + k];
                }
            }
            else {
                for (size_t k = 0; k < D_; ++k) {
                    if (k) {
                        of << ',';
                    }
                    of << '0';
                }
            }
            of << ']';
            if (j + 1 < colNums_) {
                of << ',';
            }
        }
        of << "}\n";
    }
}

SparseTensorCscFormat::SparseTensorCscFormat(size_t D, size_t rowNums, size_t colNums,
                                             std::vector<size_t>& nnzPerColVec) :
    D_(D), rowNums_(rowNums), colNums_(colNums), nnzNums_(rowNums) {
    rowIds_ = new size_t[nnzNums_]();
    nnzPerCol_ = new size_t[colNums_]();
    colRes_ = new size_t[colNums_ + 1]();
    vals_ = new double[nnzNums_ * D_]();

    assert(nnzPerColVec.size() == colNums_);
    std::copy(nnzPerColVec.begin(), nnzPerColVec.end(), nnzPerCol_);
    for (size_t i = 1; i <= colNums_; ++i) {
        colRes_[i] = colRes_[i - 1] + nnzPerCol_[i - 1];
    }
    std::iota(rowIds_, rowIds_ + nnzNums_, size_t{0});
}

SparseTensorCscFormat::SparseTensorCscFormat(SparseTensorCooFormat* coo) :
    D_(coo->D_), rowNums_(coo->rowNums_), colNums_(coo->colNums_), nnzNums_(coo->nnzNums_) {
    rowIds_ = new size_t[nnzNums_]();
    nnzPerCol_ = new size_t[colNums_]();
    colRes_ = new size_t[colNums_ + 1]();
    vals_ = new double[nnzNums_ * D_]();

    for (size_t i = 0; i < nnzNums_; ++i) {
        nnzPerCol_[coo->colIds_[i]]++;
    }
    for (size_t i = 1; i <= colNums_; ++i) {
        colRes_[i] = colRes_[i - 1] + nnzPerCol_[i - 1];
    }

    auto* colPtr = new size_t[colNums_ + 1];
    std::copy(colRes_, colRes_ + colNums_ + 1, colPtr);
    for (size_t i = 0; i < nnzNums_; ++i) {
        size_t col = coo->colIds_[i];
        size_t idx = colPtr[col]++;
        rowIds_[idx] = coo->rowIds_[i];
        for (size_t d = 0; d < D_; ++d) {
            vals_[idx * D_ + d] = coo->vals_[i * D_ + d];
        }
    }
    delete[] colPtr;
}

SparseTensorCscFormat::~SparseTensorCscFormat() {
    delete[] rowIds_;
    delete[] nnzPerCol_;
    delete[] colRes_;
    delete[] vals_;
}

void SparseTensorCscFormat::insertBatch(double* src, size_t beginPos, size_t endPos) {
    assert(beginPos <= endPos);
    std::memcpy(vals_ + beginPos * D_, src, (endPos - beginPos) * D_ * sizeof(double));
}

void SparseTensorCscFormat::display(const std::string& filename) const {
    std::ofstream of(filename);
    if (!of) {
        throw std::runtime_error("Cannot open: " + filename);
    }
    of << std::fixed << std::setprecision(3);

    std::vector<std::vector<int>> map(rowNums_, std::vector<int>(colNums_, -1));
    for (size_t c = 0; c < colNums_; ++c) {
        for (size_t j = colRes_[c]; j < colRes_[c + 1]; ++j) {
            map[rowIds_[j]][c] = static_cast<int>(j);
        }
    }

    for (size_t i = 0; i < rowNums_; ++i) {
        of << '{';
        for (size_t j = 0; j < colNums_; ++j) {
            of << '[';
            if (map[i][j] >= 0) {
                size_t ni = static_cast<size_t>(map[i][j]);
                for (size_t k = 0; k < D_; ++k) {
                    if (k) {
                        of << ',';
                    }
                    of << vals_[ni * D_ + k];
                }
            }
            else {
                for (size_t k = 0; k < D_; ++k) {
                    if (k) {
                        of << ',';
                    }
                    of << '0';
                }
            }
            of << ']';
            if (j + 1 < colNums_) {
                of << ',';
            }
        }
        of << "}\n";
    }
}

void SparseTensorCscFormat::saveToFile(std::string& baseDir, size_t fileIdx) const {
    if (!baseDir.empty() && baseDir.back() != '/') {
        baseDir += '/';
    }
    std::ostringstream oss;
    oss << baseDir << fileIdx << '_' << (fileIdx + 1);
    std::ofstream f(oss.str());
    if (!f) {
        throw std::runtime_error("Cannot open: " + oss.str());
    }
    f << std::fixed << std::setprecision(6);

    if (nnzNums_ < 100) {
        std::vector<int> bmp(rowNums_ * colNums_, -1);
        for (size_t c = 0; c < colNums_; ++c) {
            for (size_t j = colRes_[c]; j < colRes_[c + 1]; ++j) {
                bmp[rowIds_[j] * colNums_ + c] = static_cast<int>(j);
            }
        }

        for (size_t i = 0; i < rowNums_; ++i) {
            f << '{';
            for (size_t j = 0; j < colNums_; ++j) {
                f << '[';
                if (bmp[i * colNums_ + j] < 0) {
                    for (size_t k = 0; k < D_; ++k) {
                        if (k) {
                            f << ',';
                        }
                        f << '0';
                    }
                }
                else {
                    size_t ni = static_cast<size_t>(bmp[i * colNums_ + j]);
                    for (size_t k = 0; k < D_; ++k) {
                        if (k) {
                            f << ',';
                        }
                        f << vals_[ni * D_ + k];
                    }
                }
                f << "],";
            }
            f << "}\n";
        }
    }
    else {
        for (size_t c = 0; c < colNums_; ++c) {
            for (size_t j = colRes_[c]; j < colRes_[c + 1]; ++j) {
                f << c << ' ' << rowIds_[j];
                for (size_t d = 0; d < D_; ++d) {
                    f << ' ' << vals_[j * D_ + d];
                }
                f << '\n';
            }
            f << '\n';
        }
    }
}

SparseTensorCscFormat* SparseTensorConverter::convertCooToCsc(SparseTensorCooFormat* coo) {
    assert(coo != nullptr);
    auto* csc = new SparseTensorCscFormat(coo);
    assert(verifyCooEqualsCsc(coo, csc));
    for (size_t i = 0; i < csc->nnzNums_; ++i) {
        assert(csc->rowIds_[i] == i);
    }
    delete coo;
    return csc;
}

bool SparseTensorConverter::verifyCooEqualsCsc(SparseTensorCooFormat* coo, SparseTensorCscFormat* csc) {

    if (coo->D_ != csc->D_ || coo->nnzNums_ != csc->nnzNums_ || coo->rowNums_ != csc->rowNums_ ||
        coo->colNums_ != csc->colNums_) {
        return false;
    }

    const size_t R = coo->rowNums_, C = coo->colNums_, D = csc->D_;
    std::vector<std::vector<int>> cm(R, std::vector<int>(C, -1));
    std::vector<std::vector<int>> sm(R, std::vector<int>(C, -1));

    for (size_t i = 0; i < coo->nnzNums_; ++i) {
        cm[coo->rowIds_[i]][coo->colIds_[i]] = static_cast<int>(i);
    }
    for (size_t c = 0; c < C; ++c) {
        for (size_t j = csc->colRes_[c]; j < csc->colRes_[c + 1]; ++j) {
            sm[csc->rowIds_[j]][c] = static_cast<int>(j);
        }
    }

    for (size_t i = 0; i < R; ++i) {
        for (size_t j = 0; j < C; ++j) {
            if (cm[i][j] == -1 || sm[i][j] == -1) {
                if (cm[i][j] != sm[i][j]) {
                    return false;
                }
                continue;
            }
            size_t ci = static_cast<size_t>(cm[i][j]);
            size_t si = static_cast<size_t>(sm[i][j]);
            for (size_t k = 0; k < D; ++k) {
                if (!Comp::isZero(coo->vals_[ci * D + k] - csc->vals_[si * D + k])) {
                    return false;
                }
            }
        }
    }
    return true;
}
