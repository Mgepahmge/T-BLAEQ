# Memory Policies {#memory_policies}

T-BLAEQ supports four memory policies assigned per hierarchy level by the
PolicyScheduler.  This page describes the GPU data involved, how each policy
manages that data, and the scheduler algorithm that selects policies
automatically.

---

## GPU Data Classification

The data involved in a query can be classified along two independent dimensions:

- **Scale** — *large* (comparable in size to the P-tensor) vs. *small*
  (negligible fraction of device memory).
- **Type** — *known before any query* (fixed by the index structure) vs.
  *produced during a query* (depends on the specific query and pruning outcome).

| | Known before query | Produced during query |
|---|---|---|
| **Large** | P-tensor values | SpTSpM yValue output; SpTSpM index arrays; compact output values |
| **Small** | Max-radius arrays; sort-to-original maps; coarsest mesh | Pruning mask; KNN clusters; query point/bounds; compact counts and ids |

### Large-scale, known before query

| Data | Size | Role |
|------|------|------|
| P-tensor values (`dPTensorVals[l]`) | `P[l].nnz × D × 8 B` | Prolongation operator read by SpTSpM to refine each mesh level. Dominant device memory consumer. |

### Small-scale, known before query

| Data | Size | Role |
|------|------|------|
| Max-radius arrays (`dMeshMaxRadius[l]`) | `P[l].col_nums × 8 B` | One radius per centroid; used by the pruning kernel to bound cluster extent. |
| Sort-to-original maps (`dMaps[l]`) | `mesh_size[l] × 8 B` | Remaps SpTSpM output ids from sorted KMeans order back to original dataset indices. |
| Coarsest mesh (`dCoarsestMeshIds/Vals`) | `coarsest_nnz × (8 + D×8) B` | The initial input grid for every query at level 0. |

### Large-scale, produced during query

| Data | Size (worst case) | Role |
|------|-------------------|------|
| SpTSpM yValue output (`dYValue[l]`) | `P[l].nnz × D × 8 B` | Output of SpTSpM; passed as input grid to the next hierarchy level. |
| SpTSpM index arrays (`colInd`, `rowInd`, `matrixPos`) | `P[l].nnz × 24 B` | Per-nnz index arrays built on host and uploaded to drive the SpTSpM kernel. |
| Compact output values (`dCompactVals`) | `P[l].col_nums × D × 8 B` | Coordinate vectors of surviving centroids after pruning; input to SpTSpM. |

### Small-scale, produced during query

| Data | Size | Role |
|------|------|------|
| Pruning mask (`dMask`) | `p × 1 B` | Boolean selection output of the pruning kernel. |
| KNN distance results (`dClusters`) | `p × 16 B` | Per-centroid distance-label pairs; sorted then processed by the STEP algorithm. |
| Query parameters (`dQueryPoint`, `dLo`, `dHi`) | `D × 8 B` | Query point or range bounds uploaded before each pruning kernel. |
| Compact selection counts (`dProcCounts`) | `⌈p/128⌉ × 4 B` | Per-warp selected counts for the compact prefix-sum. |
| Compact output ids (`dCompactIds`) | `selectCount × 8 B` | Global ids of the surviving centroids after compaction. |

---

## Policy Descriptions

### L0 — Zero-malloc hot path

All index data (large and small, known before query) is uploaded permanently
during `prepare()`.  All working buffers for intermediate results (large and
small, produced during query) are pre-allocated once at `prepare()` time,
sized to the worst-case capacity across all levels.  Every call to `runLevel()`
reuses these pre-allocated buffers: there is no `cudaMalloc` or `cudaFree`
in the query hot path.

### L1 — Full permanent cache

All index data is uploaded permanently.  Intermediate result buffers
(SpTSpM index arrays, yValue, pruning mask, clusters, compact buffers)
are allocated and freed dynamically per query.

### L2 — Selective lazy-load

Only maps and the coarsest mesh are permanently resident.  Everything else
follows a lazy-load pattern: upload when needed, free immediately after use.

Crucially, the P-tensor is **never uploaded in full**.
`SpTSpMMultiplication_v3_L2` downloads the pruned centroid ids, collects
only the corresponding P-tensor columns from the host, and uploads that
compact subset.  When pruning is effective (typical in KNN queries), the
upload volume is a small fraction of the full P-tensor.

Max-radius is uploaded before the pruning kernel and freed immediately
after — it does not coexist on the device with the P-tensor columns.

### L3 — Tiled fallback

Nothing is permanently resident on the device.  Every operation queries
`cudaMemGetInfo` to determine the available budget, then uploads data in
tiles sized to fit, computes, downloads, and immediately frees.

Maps are never uploaded: the refactor step reads `idx_.maps[l+1]` directly
from host memory.  The coarsest mesh is never uploaded: at level 0, L3
reads `idx_.coarsestMesh` from the host pointer.

The output grid of every level is a host-resident `new[]` array
(`LevelResult::onHost == true`).

---

## Detailed Comparison

| Data / behaviour | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| P-tensor values | Permanent | Permanent | Selected columns only, transient per query | Per tile, freed after each tile |
| Max-radius arrays | Permanent | Permanent | Transient per level (freed before SpTSpM) | Per tile, pre-gathered, freed after each tile |
| Sort-to-original maps | Permanent | Permanent | Permanent | Never uploaded; host array used directly |
| Coarsest mesh | Permanent | Permanent | Permanent | Never uploaded; host pointer used directly |
| SpTSpM yValue output | Pre-allocated (`dYValue[l]`) | Dynamic per query | Dynamic per query | `new[]` on host |
| SpTSpM index arrays | Pre-allocated (`dSpTSpMBufs[l]`) | Dynamic per query | Dynamic per query | Dynamic per tile |
| Pruning mask | Pre-allocated (`dMask`) | Dynamic per query | Dynamic per query | Dynamic per tile |
| KNN clusters buffer | Pre-allocated (`dClusters`) | Dynamic per query | Dynamic per query | Dynamic per tile |
| Compact buffers | Pre-allocated | Dynamic per query | Dynamic per query | Dynamic per tile |
| Query point / bounds | Pre-allocated | Dynamic per query | Dynamic per query | Dynamic per tile |
| `cudaMalloc` in hot path | **None** | Yes | Yes | Yes (every tile) |
| Output grid residency | Device | Device | Device | **Host** |
| Tiled processing | No | No | No | **Yes** |

---

## Peak Device Memory and Scheduler

### Per-level cost estimates

Let `pVals = P[l].nnz × D × 8 B`, `radius = P[l].col_nums × 8 B`,
`spBuf = P[l].nnz × 24 B`.

| Policy | Permanent cost added to budget | Transient peak during SpTSpM |
|--------|-------------------------------|------------------------------|
| **L0** | `pVals + radius + spBuf + pVals` | None (pre-allocated) |
| **L1** | `pVals + radius` | `spBuf + pVals` |
| **L2** | None | `2 × pVals + spBuf` (radius freed before SpTSpM) |
| **L3** | None | Tile-bounded, determined at runtime |

The permanent overhead deducted before level assignment (L0, L1, L2 only):

```
permanent_overhead = deviceCoarsestMesh + deviceMaps
```

L3 contributes zero permanent overhead: it never calls `uploadPermanentData()`.

### Scheduler algorithm

```
budget = free_device_mem × 0.85 − permanent_overhead

for l = 0 .. intervals−1:        (coarsest → finest)
    if   budget ≥ costL0  →  assign L0,  budget −= costL0
    elif budget ≥ costL1  →  assign L1,  budget −= costL1
    elif budget ≥ costL2  →  assign L2   (transient: budget unchanged)
    else                  →  assign L3   (tiled:     budget unchanged)
```

L0 and L1 reduce the remaining budget permanently.
L2 and L3 leave it unchanged: their data is either transient (freed before the
next level) or tile-managed, so subsequent levels see the full remaining budget.
