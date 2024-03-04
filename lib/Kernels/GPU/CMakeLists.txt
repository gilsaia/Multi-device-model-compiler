add_library(MultiDeviceGPUOps SHARED 
Ops.cu
Utils.cu
LLM.cu
LayerNorm.cu
Transpose.cu)
target_include_directories(MultiDeviceGPUOps PRIVATE
${CUDAToolkit_INCLUDE_DIRS}
${CMAKE_SOURCE_DIR}/3rdparty/xxHash
${CMAKE_SOURCE_DIR}/3rdparty/trt_fused_multihead_attention)
target_link_directories(MultiDeviceGPUOps PRIVATE
${CUDAToolkit_LIBRARY_DIR})
target_link_libraries(MultiDeviceGPUOps PUBLIC
LLVMSupport
MultiDeviceCUDARuntime
trt_fused_multi_head_attention
)
target_link_libraries(MultiDeviceGPUOps PRIVATE
CUDA::cuda_driver
CUDA::cudart
CUDA::cublas
CUDA::cublasLt
cudnn
xxHash::xxhash)
set_property(TARGET MultiDeviceGPUOps PROPERTY CUDA_ARCHITECTURES native)