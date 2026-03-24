/**
 * @file Data_Structures.cuh
 * @brief Core data structures for the T-BLAEQ hierarchical mesh index.
 *
 * @details This file declares the fundamental data types used throughout the
 * project: PointCloud for dense host-resident datasets, SparseGrid for
 * sparse mesh levels, and SparseTensorCscFormat/CooFormat for the P-tensor
 * prolongation operators.  SparseTensorConverter converts between formats.
 *
 * Legacy typedef aliases (Multidimensional_Arr, GridAsSparseMatrix) are
 * provided for backward compatibility with code that has not been updated.
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

class SparseTensorCscFormat;
class SparseTensorCooFormat;

/**
 * @class PointCloud
 * @brief Dense host-resident array of N points in D-dimensional space.
 *
 * @details Memory layout is row-major AOS: point i occupies
 * data[i*D .. i*D+D-1].  Fully value-semantic (copyable and movable).
 * Replaces the legacy Multidimensional_Arr type.
 */
class PointCloud {
public:
    int dim = 0; //!< Number of dimensions per point.
    int size = 0; //!< Number of points.
    double* data = nullptr; //!< Owned host buffer, length = size * dim.

    PointCloud() = default;

    /*!
     * @brief Allocate an uninitialised buffer for n points of dimension d.
     *
     * @param[in] n Number of points.
     * @param[in] d Dimensionality.
     */
    PointCloud(int n, int d) : dim(d), size(n), data(new double[static_cast<size_t>(n) * d]), N(n), D(d) {}

    ~PointCloud() { delete[] data; }

    PointCloud(const PointCloud& o) :
        dim(o.dim), size(o.size), data(new double[static_cast<size_t>(o.size) * o.dim]), N(o.N), D(o.D) {
        std::copy(o.data, o.data + static_cast<size_t>(size) * dim, data);
    }

    PointCloud& operator=(const PointCloud& o) {
        if (this == &o) {
            return *this;
        }
        delete[] data;
        size = o.size;
        dim = o.dim;
        N = o.N;
        D = o.D;
        data = new double[static_cast<size_t>(size) * dim];
        std::copy(o.data, o.data + static_cast<size_t>(size) * dim, data);
        return *this;
    }

    PointCloud(PointCloud&& o) noexcept : dim(o.dim), size(o.size), data(o.data), N(o.N), D(o.D) {
        o.data = nullptr;
        o.size = 0;
        o.dim = 0;
        o.N = 0;
        o.D = 0;
    }

    PointCloud& operator=(PointCloud&& o) noexcept {
        if (this == &o) {
            return *this;
        }
        delete[] data;
        dim = o.dim;
        size = o.size;
        data = o.data;
        N = o.N;
        D = o.D;
        o.data = nullptr;
        o.size = 0;
        o.dim = 0;
        o.N = 0;
        o.D = 0;
        return *this;
    }

    /*!
     * @brief Return a const pointer to the start of point i.
     * @param[in] i Zero-based point index.
     * @return Pointer to D consecutive doubles for point i.
     */
    const double* pointAt(int i) const { return data + static_cast<size_t>(i) * dim; }

    /*!
     * @brief Return a mutable pointer to the start of point i.
     * @param[in] i Zero-based point index.
     * @return Pointer to D consecutive doubles for point i.
     */
    double* pointAt(int i) { return data + static_cast<size_t>(i) * dim; }

    int N = 0; //!< Legacy alias for size; retained for backward compatibility.
    int D = 0; //!< Legacy alias for dim; retained for backward compatibility.
};

/// @brief Backward-compatible typedef so existing call sites compile unchanged.
using Multidimensional_Arr = PointCloud;

/**
 * @class SparseGrid
 * @brief Sparse representation of one mesh level in the hierarchical index.
 *
 * @details Each non-zero entry i corresponds to one point:
 *   ids[i]                     - original (global) point index in the dataset.
 *   vals[i*dim .. i*dim+dim-1] - D-dimensional coordinate vector.
 *
 * Ownership semantics: constructors that take vectors or no data pointer own
 * their buffers and free them in the destructor.  The raw-pointer constructor
 * (size_t*, double*) wraps externally-managed arrays such as device buffers
 * returned by CUDA kernels; the destructor does NOT free those.
 *
 * Replaces the legacy GridAsSparseMatrix type.
 */
class SparseGrid {
public:
    using PointVec = std::vector<double>;

    SparseGrid() = default;

    //! Reserve structural metadata without allocating buffers.
    SparseGrid(size_t numRows, size_t dim, size_t nnz) : numRows_(numRows), dim_(dim), nnz_(nnz) {}

    /*!
     * @brief Build from a sorted list of points with an explicit row-id map.
     *
     * @param[in] points    D-dimensional point coordinates (one entry per NNZ).
     * @param[in] rowIds    Original row indices (same length as points).
     * @param[in] logicRows Total number of logical rows at this grid level.
     * @param[in] dim       Expected dimensionality (asserted against points[0].size()).
     */
    SparseGrid(const std::vector<PointVec>& points, const std::vector<size_t>& rowIds, size_t logicRows, size_t dim);

    /*!
     * @brief Build from a contiguous slice [begin, last) of a point vector.
     *
     * @details ids[i] is set to the absolute position (begin + i) in the source.
     *
     * @param[in] points Source point vector.
     * @param[in] begin  Start index (inclusive).
     * @param[in] last   End index (exclusive).
     */
    SparseGrid(const std::vector<PointVec>& points, size_t begin, size_t last);

    /*!
     * @brief Wrap externally-managed arrays without taking ownership.
     *
     * @details The caller is responsible for freeing ids and vals through
     * the appropriate API (cudaFree for device memory, delete[] for host).
     *
     * @param[in] numRows Number of logical rows.
     * @param[in] dim     Data dimensionality.
     * @param[in] nnz     Number of non-zero entries.
     * @param[in] ids     External ids array (not owned).
     * @param[in] vals    External vals array (not owned).
     */
    SparseGrid(size_t numRows, size_t dim, size_t nnz, size_t* ids, double* vals) :
        numRows_(numRows), dim_(dim), nnz_(nnz), ids_(ids), vals_(vals), ownsBuffers_(false) {}

    SparseGrid(const SparseGrid& o);
    SparseGrid& operator=(const SparseGrid& o);

    SparseGrid(SparseGrid&& o) noexcept;
    SparseGrid& operator=(SparseGrid&& o) noexcept;

    ~SparseGrid();

    bool isAos() const { return isAos_; }
    size_t numRows() const { return numRows_; }
    size_t dims() const { return dim_; }
    size_t numNnz() const { return nnz_; }
    size_t* ids() const { return ids_; }
    double* vals() const { return vals_; }

    void setAos(bool v) { isAos_ = v; }
    void setIds(size_t* p) { ids_ = p; }
    void setVals(double* p) { vals_ = p; }
    void setIdsFromVec(std::vector<size_t>& v);

    /*!
     * @brief Release owned host buffers and null the pointers.
     * @warning Only valid when ownsBuffers_ is true.
     */
    void freeDram();

    bool get_memory_arch() const { return isAos_; }
    size_t get_num_rows() const { return numRows_; }
    size_t get_dimensions() const { return dim_; }
    size_t get_nnz_nums() const { return nnz_; }
    size_t* get_ids_() const { return ids_; }
    double* get_vals_() const { return vals_; }
    void set_memory_arch(bool v) { isAos_ = v; }
    void set_ids(size_t* p) { ids_ = p; }
    void set_vals(double* p) { vals_ = p; }
    void set_ids_using_vec(std::vector<size_t>& v) { setIdsFromVec(v); }
    void set_nnz_vals_P_col_ids_using_vec(std::vector<size_t>&) {}
    void pre_allocate_d_vals() {}
    void pre_allocate_h_vals() {}
    void free_DRAM() { freeDram(); }
    void Load_Coreast_Mesh_to_DRAM() {}

private:
    bool isAos_ = true; //!< True when values are in AOS layout.
    size_t numRows_ = 0; //!< Total logical rows at this grid level.
    size_t dim_ = 0; //!< Data dimensionality.
    size_t nnz_ = 0; //!< Number of non-zero entries.
    size_t* ids_ = nullptr; //!< Point ids array (host or device).
    double* vals_ = nullptr; //!< Point value array (host or device).
    bool ownsBuffers_ = true; //!< False when wrapping external arrays.
};

using GridAsSparseMatrix = SparseGrid;

/**
 * @class SparseTensorCooFormat
 * @brief 3-D sparse tensor in COO (coordinate) format.
 *
 * @details Each non-zero stores a row coordinate, a column coordinate and a
 * D-dimensional value vector.  Used as an intermediate format during index
 * construction; converted to CSC via SparseTensorConverter::convertCooToCsc().
 */
class SparseTensorCooFormat {
public:
    friend class SparseTensorCscFormat;
    friend class SparseTensorConverter;

    using PointVec = std::vector<double>;

    /*!
     * @brief Construct an empty COO tensor with the given dimensions.
     *
     * @param[in] D       Depth (dimensionality of each value vector).
     * @param[in] rowNums Number of logical rows.
     * @param[in] colNums Number of logical columns.
     */
    explicit SparseTensorCooFormat(size_t D, size_t rowNums, size_t colNums);
    ~SparseTensorCooFormat();

    SparseTensorCooFormat(const SparseTensorCooFormat&) = delete;
    SparseTensorCooFormat& operator=(const SparseTensorCooFormat&) = delete;

    /*!
     * @brief Append one non-zero entry.
     *
     * @param[in] row    Row coordinate.
     * @param[in] col    Column coordinate.
     * @param[in] valVec Pointer to D consecutive values.
     */
    void insertNnz(size_t row, size_t col, double* valVec);

    /// @brief Debug: dump as a dense matrix to @p filename.
    void display(const std::string& filename) const;

    void insert_one_nnz(size_t r, size_t c, double* v) { insertNnz(r, c, v); }

private:
    size_t D_;
    size_t writeIdx_ = 0;
    size_t rowNums_;
    size_t colNums_;
    size_t nnzNums_;
    size_t* rowIds_;
    size_t* colIds_;
    double* vals_;
};

/**
 * @class SparseTensorCscFormat
 * @brief 3-D sparse tensor in CSC (Compressed Sparse Column) format.
 *
 * @details Stores the prolongation operator P at one hierarchy level.
 * Columns correspond to centroids (coarse-mesh points); rows correspond to
 * fine-mesh points.  Because each fine point belongs to exactly one centroid,
 * P has exactly row_nums non-zeros (one per row).  Each non-zero value is a
 * D-dimensional ratio vector of fine-mesh to coarse-mesh coordinates.
 *
 * The CSC layout enables efficient column-wise access during SpTSpM:
 * colRes[c] .. colRes[c+1] gives the nnz range for centroid c.
 */
class SparseTensorCscFormat {
public:
    friend class SparseTensorConverter;

    /*!
     * @brief Construct an empty CSC tensor with the given dimensions.
     *
     * @param[in] D       Depth (dimensionality of each value vector).
     * @param[in] rowNums Number of rows (fine-mesh point count).
     * @param[in] colNums Number of columns (centroid count).
     */
    explicit SparseTensorCscFormat(size_t D, size_t rowNums, size_t colNums) :
        D_(D), rowNums_(rowNums), colNums_(colNums) {}

    /*!
     * @brief Construct and pre-allocate storage from per-column NNZ counts.
     *
     * @param[in] D           Depth.
     * @param[in] rowNums     Row count.
     * @param[in] colNums     Column count.
     * @param[in] nnzPerColVec NNZ count per column (length must equal colNums).
     */
    explicit SparseTensorCscFormat(size_t D, size_t rowNums, size_t colNums, std::vector<size_t>& nnzPerColVec);

    /*!
     * @brief Construct by converting from COO format.
     *
     * @param[in] coo Source COO tensor (not consumed; caller still owns it).
     */
    explicit SparseTensorCscFormat(SparseTensorCooFormat* coo);

    ~SparseTensorCscFormat();

    SparseTensorCscFormat(const SparseTensorCscFormat&) = delete;
    SparseTensorCscFormat& operator=(const SparseTensorCscFormat&) = delete;

    bool getMemoryArch() const { return isAos_; }
    size_t getDim() const { return D_; }
    size_t getRowNums() const { return rowNums_; }
    size_t getColNums() const { return colNums_; }
    size_t getNnzNums() const { return nnzNums_; }
    const size_t* getRowIds() const { return rowIds_; }
    const size_t* getNnzPerCol() const { return nnzPerCol_; }
    const size_t* getColRes() const { return colRes_; }
    const double* getVals() const { return vals_; }

    /// @brief Non-const accessors -- required by serializer and kernel wrappers.
    size_t* getRowIdsMut() { return rowIds_; }
    size_t* getColResMut() { return colRes_; }
    double* getValsMut() { return vals_; }

    bool get_memory_arch() const { return isAos_; }
    size_t get_dim() const { return D_; }
    size_t get_row_nums() const { return rowNums_; }
    size_t get_col_nums() const { return colNums_; }
    size_t get_nnz_nums() const { return nnzNums_; }
    const size_t* get_row_ids() const { return rowIds_; }
    const size_t* get_nnz_per_col() const { return nnzPerCol_; }
    const size_t* get_col_res() const { return colRes_; }
    const double* get_vals() const { return vals_; }


    /*!
     * @brief Bulk-copy a range of NNZ values into the internal value buffer.
     *
     * @param[in] src      Source array; length must be (endPos - beginPos) * D.
     * @param[in] beginPos Start NNZ index.
     * @param[in] endPos   End NNZ index (exclusive).
     */
    void insertBatch(double* src, size_t beginPos, size_t endPos);

    void Insert_One_Batch(double* s, size_t b, size_t e) { insertBatch(s, b, e); }

    /// @brief Debug: dump the tensor as a text file.
    void display(const std::string& filename) const;

    /// @brief Write index content to a human-readable file for inspection.
    void saveToFile(std::string& baseDir, size_t fileIdx) const;

    void Load_To_File(std::string& d, size_t i) { saveToFile(d, i); }

private:
    bool isAos_ = true; //!< True when vals are in AOS layout.
    size_t D_; //!< Value vector dimensionality.
    size_t rowNums_; //!< Number of rows (fine-mesh point count).
    size_t colNums_; //!< Number of columns (centroid count).
    size_t nnzNums_ = 0; //!< Total number of non-zeros.
    size_t* rowIds_ = nullptr; //!< Row index per nnz (fine-point global id).
    size_t* nnzPerCol_ = nullptr; //!< NNZ count per column.
    size_t* colRes_ = nullptr; //!< Column offset array (length colNums + 1).
    double* vals_ = nullptr; //!< Value array in AOS layout (nnzNums * D).
};

/**
 * @class SparseTensorConverter
 * @brief Utility class for converting between COO and CSC sparse-tensor formats.
 */
class SparseTensorConverter {
public:
    /*!
     * @brief Convert a COO tensor to CSC format and free the source.
     *
     * @details Asserts structural equality between source and result as a
     * correctness check.
     *
     * @param[in] coo Heap-allocated COO tensor; ownership is consumed and freed.
     * @return Newly allocated CSC tensor. Caller owns.
     */
    static SparseTensorCscFormat* convertCooToCsc(SparseTensorCooFormat* coo);

    /*!
     * @brief Check that a COO tensor and a CSC tensor store identical non-zeros.
     *
     * @param[in] coo Source COO tensor.
     * @param[in] csc Target CSC tensor to compare against.
     * @return True when all non-zeros match in value and position.
     */
    static bool verifyCooEqualsCsc(SparseTensorCooFormat* coo, SparseTensorCscFormat* csc);

    static SparseTensorCscFormat* Convert_Coo2Csc(SparseTensorCooFormat* c) { return convertCooToCsc(c); }
};
