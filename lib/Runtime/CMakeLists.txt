add_library(MultiDeviceRuntime STATIC ModelInfo.cpp RuntimeUtil.cpp)

target_link_libraries(MultiDeviceRuntime PUBLIC
${CMAKE_DL_LIBS}
LLVMSupport)
add_subdirectory(TPU)
if(USE_CUDA_SUPPORT)
    add_subdirectory(CUDA)
endif()