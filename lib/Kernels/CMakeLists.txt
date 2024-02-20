add_subdirectory(CPU)
if(USE_CUDA_SUPPORT)
    add_subdirectory(GPU)
endif()