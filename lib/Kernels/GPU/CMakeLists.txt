add_library(MultiDeviceGPUOps SHARED Ops.cu)
target_include_directories(MultiDeviceGPUOps PRIVATE
${CUDAToolkit_INCLUDE_DIRS})
target_link_directories(MultiDeviceGPUOps PRIVATE
${CUDAToolkit_LIBRARY_DIR})
target_link_libraries(MultiDeviceGPUOps PUBLIC
LLVMSupport
)
target_link_libraries(MultiDeviceGPUOps PRIVATE
CUDA::cuda_driver
CUDA::cudart
CUDA::cublas
CUDA::cublasLt)