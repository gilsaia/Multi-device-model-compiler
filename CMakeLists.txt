cmake_minimum_required(VERSION 3.18)

project(multi-device-compiler VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>/bin)
set(CMAKE_INCLUDE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>/include)

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

# ONNX-Mlir Configure
set(ONNX_MLIR_CCACHE_BUILD ON)
add_subdirectory(3rdparty/onnx-mlir)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/onnx-mlir/include)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/onnx-mlir/src)
include_directories(${CMAKE_BINARY_DIR}/3rdparty/onnx-mlir/src)

#---------------------------------------------------------------

include_directories(${CMAKE_SOURCE_DIR}/include)

add_subdirectory(tools)
add_subdirectory(lib)