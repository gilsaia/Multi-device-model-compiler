add_llvm_executable(multi-device-gpu-runner multi-device-gpu-runner.cpp)

target_link_libraries(multi-device-gpu-runner PUBLIC
${CMAKE_DL_LIBS}
MultiDeviceRuntime
MultiDeviceCUDARuntime
LLVMSupport)