/**
 * @file IndexSerializer.cuh
 * @brief Saves and loads the hierarchical mesh index to and from disk.
 *
 * @details IndexSerializer reads and writes a directory that contains one
 * binary file per index component.  The file naming convention is fixed so
 * that the directory can be identified and loaded unambiguously:
 *   metadata.bin        - structural parameters (D, N, height, intervals, ratios)
 *   Coreast_Mesh.bin    - coarsest mesh SparseGrid
 *   P_Tensor_<i>.bin    - P-tensor for level i in CSC format
 *   MaxRadius_<i>.bin   - per-centroid max-radius array for level i
 *   Map_<i>.bin         - sort-to-original map for level i
 */

#pragma once

#include <string>
#include "IndexData.cuh"

/**
 * @class IndexSerializer
 * @brief Static utility class for index persistence.
 *
 * @details All file I/O for the index is centralised here to keep IndexData
 * free of any persistence logic.  Both save() and load() operate on the
 * complete index; there is no partial or incremental serialisation.
 * The loaded IndexData contains only host-side data, exactly as if it had
 * been produced by IndexBuilder.
 */
class IndexSerializer {
public:
    /*!
     * @brief Save a fully-built IndexData to a directory on disk.
     *
     * @details Writes metadata, coarsest mesh, and all per-level files.
     * The directory must already exist; individual files are overwritten
     * if they are present.
     *
     * @param[in] data    The index to serialise.
     * @param[in] dirPath Path to the output directory.
     */
    static void save(const IndexData& data, const std::string& dirPath);

    /*!
     * @brief Load an IndexData from a directory on disk.
     *
     * @details Reads all component files and reconstructs the host-side
     * index.  The returned object is heap-allocated; the caller takes
     * ownership.  No device data is uploaded; that is deferred to the
     * strategy's prepare() call.
     *
     * @param[in] dirPath Path to the directory containing the index files.
     * @return Heap-allocated IndexData containing the fully loaded host-side index.
     */
    static IndexData* load(const std::string& dirPath);

private:
    static void saveMetadata(const IndexData& d, const std::string& path);
    static void loadMetadata(IndexData& d, const std::string& path);

    static void saveSparseTensorCsc(const SparseTensorCscFormat* t, const std::string& path);
    static SparseTensorCscFormat* loadSparseTensorCsc(const std::string& path);

    static void saveMaxRadius(const double* radius, size_t count, const std::string& path);
    static double* loadMaxRadius(const std::string& path, size_t& countOut);

    static void saveMap(const size_t* map, size_t size, const std::string& path);
    static size_t* loadMap(const std::string& path, size_t& sizeOut);

    static void saveCoarsestMesh(const GridAsSparseMatrix* mesh, const std::string& path);
    static GridAsSparseMatrix* loadCoarsestMesh(const std::string& path);
};
