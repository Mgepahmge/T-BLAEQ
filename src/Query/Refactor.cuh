/**
 * @file Refactor.cuh
 * @brief In-place remapping of SpTSpM output ids via a sort-to-original map.
 *
 * @details After SpTSpM produces output ids in sorted order, refactor()
 * translates them back to original dataset indices using the map array
 * stored in IndexData::dMaps (device) or IndexData::maps (host).
 */

#ifndef CUDADB_REFACTOR_CUH
#define CUDADB_REFACTOR_CUH

#include "src/Data_Structures/Data_Structures.cuh"

/*!
 * @brief Remap grid ids in-place from sorted order to original dataset indices.
 *
 * @details Launches a kernel that replaces each entry grid.ids_[i] with
 * map[grid.ids_[i]].  Both grid.ids_ and map must reside in device memory.
 * Used as the final step of every runLevel() call in all four strategies.
 *
 * @param[in,out] grid The SparseGrid whose ids_ are remapped in-place (device).
 * @param[in]     map  Sort-to-original index map (device), length >= max(ids_).
 */
void refactor(GridAsSparseMatrix& grid, const size_t* map);

#endif // CUDADB_REFACTOR_CUH
