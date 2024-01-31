add_library(MultiDeviceTPURuntime SHARED TPURuntimeWrappers.cpp)
set(TPU_DIR ${CMAKE_SOURCE_DIR}/3rdparty/tpu-mlir/third_party/nntoolchain)
target_include_directories(MultiDeviceTPURuntime PRIVATE
${TPU_DIR}/include)
target_link_directories(MultiDeviceTPURuntime PRIVATE
${TPU_DIR}/lib)
target_link_libraries(MultiDeviceTPURuntime PRIVATE
MultiDeviceRuntime
bmrt
cmodel
bmlib)
target_compile_definitions(MultiDeviceTPURuntime PRIVATE USING_CMODEL)