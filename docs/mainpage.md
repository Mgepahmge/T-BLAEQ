# T-BLAEQ {#mainpage}

**T-BLAEQ** (Tile-Based Layered Approximate Exact Query) is a GPU-accelerated
hierarchical mesh index designed for fast range and K-nearest-neighbour (KNN)
queries over large high-dimensional datasets.

T-BLAEQ organises the dataset into a coarsening hierarchy of meshes connected
by sparse prolongation tensors (P-tensors).  Each query traverses the hierarchy
from coarsest to finest, pruning irrelevant clusters at every level before
expanding survivors via a sparse matrix-vector multiply (SpTSpM).  A greedy
memory-policy scheduler automatically decides how aggressively to cache index
data on the GPU, adapting to whatever device memory is available.

For a detailed description of the memory policies and the query pipeline,
see the @subpage memory_policies and @subpage query_pipeline pages.

---

## Requirements

### Hardware

- NVIDIA GPU with CUDA Compute Capability 7.0 or higher (Volta+)

### Software

- **CUDA Toolkit**: 12.0 or higher
- **GCC / G++**: 12.0 or higher
- **CMake**: 3.20 or higher
- **cuVS / RAFT**: CUDA Vector Search library — [https://github.com/rapidsai/cuvs](https://github.com/rapidsai/cuvs)
- **Doxygen** *(optional, for documentation)*: 1.9 or higher
- **Graphviz** *(optional, for call/class graphs)*: any recent version

---

## Quick Start

### Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Run a range query

```bash
./T-BLAEQ --test-query \
    -i indexes/my_index \
    -f queries/range_50.txt \
    -t 0 \
    -r results.csv
```

### Run a KNN query

```bash
./T-BLAEQ --test-query \
    -i indexes/my_index \
    -f queries/knn_points.txt \
    -t 1 \
    -r results.csv \
    --K 10
```

### Force a specific memory policy (for benchmarking)

```bash
./T-BLAEQ --test-query -i indexes/my_index -f queries/range_50.txt \
    -t 0 -r results.csv --force-policy L1
```

### API usage

```cpp
// Load a pre-built index and run a range query
QueryHandler handler("indexes/my_index/", /*loadFromIndex=*/true);
handler.prepareForQuery();   // auto-selects memory policy

QueryResult result = handler.performQuery(
    "queries/range_50.txt", QueryType::RANGE);

saveQueryResult(result, "output.csv");
```

---

## Architecture

```
QueryHandler          public facade: build/load, prepare, query
    └── QueryEngine   outer query loop, timing, inter-level grid lifetime
            └── IQueryStrategy   per-level strategy interface
                    ├── L0Strategy   zero cudaMalloc hot path
                    ├── L1Strategy   full permanent cache
                    ├── L2Strategy   selective lazy-load
                    └── L3Strategy   tiled fallback
```

**IndexData** is the central data container.  It owns all host-side index
arrays and all device buffers whose lifetimes are managed by the active
strategy.

**PolicyScheduler** analyses available device memory, assigns the most
aggressive feasible policy to each hierarchy level, and instantiates the
appropriate strategy object.

---

## Directory Structure

```
src/
  core/               IndexData, QueryEngine, QueryHandler, MemoryPolicy
    strategies/       L0–L3 strategy implementations and shared helpers
  Data_Structures/    PointCloud, SparseGrid, SparseTensorCscFormat, Query
  Kernel/             SpMSpV kernels and strategy-specific wrappers
  Kmeans/             GPU KMeans clustering (RAFT/cuVS wrapper)
  MergeSort/          Block merge sort kernel and host merge utilities
  Query/              Pruning kernels, compact, refactor, CUDA error macros
  Setup/              Index construction helpers (P-tensor, max-radius)
  utils/              NVTXProfiler, path string utilities
docs/
  mainpage.md         This page
  memory_policies.md  GPU data classification and memory policy details
  query_pipeline.md   Per-step query pipeline description
  Doxyfile            Doxygen configuration
  buildDoc.sh         Documentation build script
```

---

## Building the Documentation

```bash
cd docs
./buildDoc.sh
# Output: docs/html/index.html
```
