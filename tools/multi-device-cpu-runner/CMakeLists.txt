add_llvm_executable(multi-device-cpu-runner multi-device-cpu-runner.cpp)

target_link_libraries(multi-device-cpu-runner PUBLIC
${CMAKE_DL_LIBS}
MultiDeviceRuntime
LLVMSupport)