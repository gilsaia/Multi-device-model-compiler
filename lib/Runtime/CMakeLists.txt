add_library(MultiDeviceRuntime STATIC ModelInfo.cpp RuntimeUtil.cpp)

target_link_libraries(MultiDeviceRuntime PUBLIC
${CMAKE_DL_LIBS}
LLVMSupport)

if(USE_CUDA_SUPPORT)
    add_subdirectory(CUDA)
endif()