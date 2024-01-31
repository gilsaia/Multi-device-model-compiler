add_llvm_executable(multi-device-tpu-runner multi-device-tpu-runner.cpp)

target_link_libraries(multi-device-tpu-runner PUBLIC
MultiDeviceTPURuntime
LLVMSupport)