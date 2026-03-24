# T-BLAEQ

**T-BLAEQ** (Tile-Based Layered Approximate Exact Query) is the official implementation for our submitted paper to VLDB 2026 titled:  
*"T-BLAEQ: A Tensor-Based Multigrid Index for GPU-Accelerated Multi-dimensional Query Processing"*

The code is fully implemented in CUDA C/C++.

## Overview

T-BLAEQ is a GPU-accelerated hierarchical mesh index designed for fast range and K-nearest-neighbour (KNN) queries over large high-dimensional datasets. 

It organises the dataset into a coarsening hierarchy of meshes connected by sparse prolongation tensors (P-tensors). Each query traverses the hierarchy from coarsest to finest, pruning irrelevant clusters at every level before expanding survivors via a sparse matrix-vector multiply (SpTSpM). A **greedy memory-policy scheduler** automatically decides how aggressively to cache index data on the GPU, adapting to whatever device memory is available.

## Requirements

### Hardware
- NVIDIA GPU with CUDA Compute Capability 7.0 or higher (Volta+)

### Software
- **CUDA Toolkit**: 12.0 or higher
- **GCC / G++**: 12.0 or higher
- **CMake**: 3.20 or higher
- **cuVS / RAFT**: CUDA Vector Search library ([https://github.com/rapidsai/cuvs](https://github.com/rapidsai/cuvs))
- **Doxygen** *(optional, for building documentation)*: 1.9 or higher
- **Graphviz** *(optional, for call/class graphs)*: any recent version

## Building

To build the project, use CMake. It is highly recommended to build in `Release` mode for optimal GPU performance:

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Usage

### Command-Line Interface (CLI)

After building, run the program from the build directory:
```bash
./T-BLAEQ [OPTIONS]
```

#### Key Options
- `--build-index` - Build and save index for the specified dataset
- `--test-query` - Test queries on the specified dataset
- `-d, --dataset TEXT` - Path to the dataset file
- `-i, --index-path TEXT` - Path to save/load index (default: `indexes/`)
- `-f, --query-file TEXT` - Path to the query file
- `-q, --max-queries INT` - Maximum number of queries to process (default: 10)
- `-t, --query-type INT` - Type of query to test: `0` for Range Query, `1` for KNN Query
- `-k, --K INT` - Number of neighbors for KNN query (default: 10)
- `-r TEXT` - Path to save query results (e.g., `results.csv`)
- `--force-policy TEXT` - Force a specific memory policy for benchmarking (e.g., `L0`, `L1`, `L2`, `L3`)

#### CLI Examples

**Build an index:**
```bash
./T-BLAEQ --build-index -d dataset.txt -i indexes/my_index/
```

**Run range queries:**
```bash
./T-BLAEQ --test-query -i indexes/my_index/ -f queries/range_50.txt -t 0 -q 100 -r results.csv
```

**Run KNN queries:**
```bash
./T-BLAEQ --test-query -i indexes/my_index/ -f queries/knn_points.txt -t 1 --K 20 -q 100 -r results.csv
```

**Force a specific memory policy (for benchmarking):**
```bash
./T-BLAEQ --test-query -i indexes/my_index/ -f queries/range_50.txt -t 0 -r results.csv --force-policy L1
```

### C++ API Usage

You can also integrate T-BLAEQ directly into your own C++ applications:

```cpp
#include "QueryHandler.h"

// Load a pre-built index and run a range query
QueryHandler handler("indexes/my_index/", /*loadFromIndex=*/true);

// Auto-selects the optimal memory policy based on available GPU VRAM
handler.prepareForQuery();   

// Execute query
QueryResult result = handler.performQuery("queries/range_50.txt", QueryType::RANGE);

saveQueryResult(result, "output.csv");
```

## Architecture Overview

```text
QueryHandler          (Public facade: build/load, prepare, query)
    └── QueryEngine   (Outer query loop, timing, inter-level grid lifetime)
            └── IQueryStrategy   (Per-level strategy interface)
                    ├── L0Strategy   (Zero cudaMalloc hot path)
                    ├── L1Strategy   (Full permanent cache)
                    ├── L2Strategy   (Selective lazy-load)
                    └── L3Strategy   (Tiled fallback)
```

- **IndexData** is the central data container. It owns all host-side index arrays and all device buffers whose lifetimes are managed by the active strategy.
- **PolicyScheduler** analyses available device memory, assigns the most aggressive feasible policy to each hierarchy level, and instantiates the appropriate strategy object.

## Directory Structure

```text
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
docs/                 Doxygen configuration and Markdown documentation pages
```

## Documentation

To build the full Doxygen documentation (including Memory Policies and Query Pipeline details):

```bash
cd docs
./buildDoc.sh
```
The generated documentation will be available at `docs/html/index.html`.

## License

See [LICENSE](LICENSE) file for details.

## Third-Party Licenses

This project uses third-party libraries. See [NOTICE](NOTICE) and the [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES/) directory for details.