cmake_minimum_required(VERSION 3.20)
list(APPEND CMAKE_PREFIX_PATH "/opt/miniconda3")

project(T-BLAEQ CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES native)

link_directories(BEFORE /opt/miniconda3/lib)
add_link_options("-Wl,--disable-new-dtags")
add_link_options("-L/opt/miniconda3/lib")

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP      REQUIRED)
find_package(cuvs        REQUIRED)

# Core layer
set(CORE_SOURCES
    src/core/IndexData.cu
    src/core/IndexBuilder.cu
    src/core/IndexSerializer.cu
    src/core/MemoryPolicy.cu
    src/core/QueryEngine.cu
    src/core/QueryHandler.cu
    src/core/strategies/StrategyCommon.cu
    src/core/strategies/L0Strategy.cu
    src/core/strategies/L1Strategy.cu
    src/core/strategies/L2Strategy.cu
    src/core/strategies/L3Strategy.cu
)

# Data structures
set(DS_SOURCES
    src/Data_Structures/Data_Structures.cu
    src/Data_Structures/File.cu
)

# Kernels
set(KERNEL_SOURCES
    src/Kernel/SpMSpV.cu
)

# Mesh / sort
set(MERGE_SORT_SOURCES
    src/MergeSort/MergeSort.cu
)

# Query-stage kernels
set(QUERY_KERNEL_SOURCES
    src/Query/GridCompact.cu
    src/Query/KNN.cu
    src/Query/Refactor.cu
    src/Query/RangePruning.cu
    src/Query/L3Kernels.cu
)

# Setup utilities
set(SETUP_SOURCES
    src/Setup/Setup.cu
)

# KMeans
set(KMEANS_CUDA_SOURCES
    src/Kmeans/CUDAKmeans.cu
)

# Utilities
set(UTIL_SOURCES
    src/utils/NVTXProfiler.cu
    src/utils/Utils.cu
)

add_executable(T-BLAEQ
    src/main.cu
    ${CORE_SOURCES}
    ${DS_SOURCES}
    ${KERNEL_SOURCES}
    ${MERGE_SORT_SOURCES}
    ${QUERY_KERNEL_SOURCES}
    ${SETUP_SOURCES}
    ${KMEANS_CUDA_SOURCES}
    ${UTIL_SOURCES}
)

target_include_directories(T-BLAEQ PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(T-BLAEQ PRIVATE
    CUDA::cudart_static
    CUDA::cusparse
    CUDA::cublas
    OpenMP::OpenMP_CXX
    cuvs::cuvs
)

target_compile_options(T-BLAEQ PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --extended-lambda
        -Xcompiler=-Wall
        -Xcompiler=-fopenmp
    >
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
