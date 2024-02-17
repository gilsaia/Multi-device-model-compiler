add_llvm_executable(multi-device-gpu-runner multi-device-gpu-runner.cpp)

target_include_directories(multi-device-gpu-runner PUBLIC
${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(multi-device-gpu-runner PUBLIC
${CMAKE_DL_LIBS}
MultiDeviceRuntime
MultiDeviceCUDARuntime
MultiDeviceGPUOps
LLVMSupport)