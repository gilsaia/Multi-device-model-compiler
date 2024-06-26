cmake_minimum_required(VERSION 3.18)

project(multi-device-compiler VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>/bin)
set(CMAKE_INCLUDE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>/include)

option(USE_CCACHE_BUILD "Use ccache to speedup." ON)
option(USE_OMP "Use omp." ON)
option(USE_CUDA_SUPPORT "Use cuda." OFF)
option(ONLY_BUILD_PLAIN_RUNTIME "Use to build runtime without anything" OFF)

#---------------------------------------------------------------

# LLVM Configure
set(LLVM_DIR ${CMAKE_SOURCE_DIR}/3rdparty/llvm-project/build/lib/cmake/llvm)
set(MLIR_DIR ${CMAKE_SOURCE_DIR}/3rdparty/llvm-project/build/lib/cmake/mlir)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

add_definitions(${LLVM_DEFINITIONS})
#---------------------------------------------------------------

# Plain runtime build
if(ONLY_BUILD_PLAIN_RUNTIME)
    message(STATUS "Will only build plain runtime,not import other repo.")
    add_compile_definitions(MULTI_DEVICE_PLAIN_RUNTIME)
else()
#---------------------------------------------------------------

# ONNX-Mlir Configure
if (USE_CCACHE_BUILD)
    set(ONNX_MLIR_CCACHE_BUILD ON)
endif()
add_subdirectory(3rdparty/onnx-mlir)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/onnx-mlir/include)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/onnx-mlir)
include_directories(${CMAKE_BINARY_DIR}/3rdparty/onnx-mlir)

#---------------------------------------------------------------

# flat buffers
set(FLATBUFFERS_BUILD_TESTS OFF)
add_subdirectory(3rdparty/flatbuffers)
include_directories(3rdparty/flatbuffers/include)

# oneDNN
set(DNNL_CPU_RUNTIME OMP)
set(DNNL_BUILD_EXAMPLES OFF)
set(DNNL_BUILD_TESTS OFF)
set(ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_BACKEND ON)
add_subdirectory(3rdparty/oneDNN)
include_directories(3rdparty/oneDNN/include)
include_directories(${CMAKE_BINARY_DIR}/3rdparty/oneDNN/include)

# xxHash
add_subdirectory(3rdparty/xxHash/cmake_unofficial EXCLUDE_FROM_ALL)

# TPU-mlir Configure
add_subdirectory(3rdparty/tpu-mlir)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/tpu-mlir/include)
include_directories(${CMAKE_BINARY_DIR}/3rdparty/tpu-mlir/include)
include_directories(${TPU_MLIR_THIRDPARTY_INCLUDE})
link_directories(${TPU_MLIR_THIRDPARTY_LIB})
#---------------------------------------------------------------

# OMP Configure
if(USE_OMP)
    find_package(OpenMP)
    if(OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()
#---------------------------------------------------------------

# CUDA Configure
if(USE_CUDA_SUPPORT)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(MULTI_DEVICE_CUDA_ENABLE)
endif()
#---------------------------------------------------------------

# CUDA 3rdparty Kernels Configure
if(USE_CUDA_SUPPORT)
    add_subdirectory(3rdparty/trt_fused_multihead_attention)
endif()
#---------------------------------------------------------------

# end plain runtime
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

add_subdirectory(include/multi-device-model-compiler)
add_subdirectory(lib)
add_subdirectory(tools)
if(NOT ONLY_BUILD_PLAIN_RUNTIME)
    add_subdirectory(test)
endif()