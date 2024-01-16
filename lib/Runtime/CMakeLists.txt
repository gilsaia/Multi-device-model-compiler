add_library(MultiDeviceRuntime STATIC ModelInfo.cpp)

target_link_libraries(MultiDeviceRuntime PUBLIC
LLVMSupport)

if(USE_CUDA_SUPPORT)
    add_subdirectory(CUDA)
endif()