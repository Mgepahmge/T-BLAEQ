# Query Pipeline {#query_pipeline}

This page describes the four steps executed at every hierarchy level during
a T-BLAEQ query, and explains how the active memory policy affects each step.

---

## Overview

A query starts at the coarsest mesh level (level 0) and descends to the
finest (level `intervals − 1`).  At each level the current grid (a sparse
set of centroid coordinates and ids) is progressively refined:

```
coarsestMesh
    │
    ▼  [level 0]  Prune → Compact → SpTSpM → Refactor
    │
    ▼  [level 1]  Prune → Compact → SpTSpM → Refactor
    │
    ▼  [level 2]  Prune → Compact → SpTSpM → Refactor
    │
    ▼
  fine-mesh result  (returned to the caller)
```

---

## Step 1 — Pruning

**Goal**: eliminate centroids that cannot contribute to the query answer.

### Range pruning

For each centroid `c_i` with radius `r_i`, the kernel checks whether the
sphere of radius `r_i` centred at `c_i` intersects the query box.
Centroids whose sphere does not overlap are masked out.

### KNN pruning (STEP algorithm)

1. `calculateClusterDistanceKernel` computes the lower-bound distance from
   the query point to each centroid: `dist(query, centroid) − radius`.
2. The resulting `Cluster` array (distance + label) is sorted on device
   using `blockMergeSortKernel` followed by `hostSerialMergeSort`.
3. The STEP algorithm scans sorted clusters on the host, accumulating
   fine-point counts until K neighbours are guaranteed to be covered,
   then stops.

**Policy impact**:
- **L0**: radius pre-uploaded; mask, cluster, and query-point buffers
  pre-allocated; zero `cudaMalloc`.
- **L1**: radius pre-uploaded; mask and cluster buffers allocated per query.
- **L2**: radius uploaded before this step and freed immediately after.
- **L3**: radius and centroids uploaded per tile (pre-gathered to avoid
  indirect device addressing); block-sorted on device, downloaded, merged
  and STEP-processed on host.

---

## Step 2 — Compact

**Goal**: collect the surviving centroids into a contiguous device array.

The compaction uses a three-stage warp-parallel prefix-sum pipeline:

1. `launchCountKernel` — count selected elements per warp.
2. `launchPrefixSumKernel` — compute inclusive prefix-sum of warp counts.
3. `gridCompactPrefixKernel` — scatter selected entries to contiguous output.

The output is a new `SparseGrid` containing only the pruned centroids.

**Policy impact**:
- **L0**: all compaction buffers (mask upload, proc counts, output ids/vals)
  pre-allocated; zero `cudaMalloc`.
- **L1/L2**: compaction buffers allocated per query.
- **L3**: compaction is performed entirely on the host after the pruning
  mask is downloaded; no device allocation needed.

---

## Step 3 — SpTSpM

**Goal**: multiply the pruned centroid vector by the P-tensor to obtain the
refined fine-mesh output for the next level.

The kernel `SpMSpVKernelAOS_v2` computes:

```
yValue[i] = P_vals[matrixPos[i]] × centroid_vals[colInd[i]]
```

where `colInd`, `rowInd`, and `matrixPos` are pre-built index arrays that
map each output element to its P-tensor column and input centroid.

**Policy impact**:
- **L0** (`SpTSpMMultiplication_v3_L0_nb`): P-tensor values, all index
  arrays, and the yValue output buffer are pre-allocated; zero `cudaMalloc`.
- **L1** (`SpTSpMMultiplication_v3`): P-tensor values permanently resident;
  index arrays and yValue allocated per query.
- **L2** (`SpTSpMMultiplication_v3_L2`): only the P-tensor columns
  corresponding to pruned centroids are uploaded (collected from the host
  with per-column `memcpy`); index arrays and yValue allocated per query.
- **L3** (`runSpTSpMTiled`): P-tensor processed in column tiles; each tile
  uploads P-vals and index arrays, runs the kernel, downloads the partial
  output, and frees immediately.

---

## Step 4 — Refactor

**Goal**: remap SpTSpM output ids from the KMeans-sorted order back to
original dataset indices.

Each output id `yIds[i]` is replaced by `map[l+1][yIds[i]]`.

**Policy impact**:
- **L0/L1/L2**: the map `dMaps[l+1]` is permanently resident on the device;
  the refactor kernel runs in-place.
- **L3**: the map `idx_.maps[l+1]` is read directly from the host array;
  refactor is a simple host loop with no device involvement.

---

## Output and Ownership

The `LevelResult` returned by each `runLevel()` call carries three flags
that tell `QueryEngine` how to manage the returned grid:

| Flag | Meaning |
|------|---------|
| `ownsIds` | `false` for L0: `ids_` points into `dSpTSpMBufs[l].rowInd` owned by `IndexData`. |
| `ownsVals` | `false` for L0: `vals_` points into `l0Bufs.dYValue[l]` owned by `IndexData`. |
| `onHost` | `true` for L3: `ids_` and `vals_` are `new[]`-allocated host arrays; free with `delete[]`. |

`QueryEngine` uses these flags to choose the correct deallocation path
between levels, preventing both leaks and double-frees.
