#include "multi-device-model-compiler/Kernels/GPU/Utils.h"

#include "cuda_fp16.h"

template <size_t bytes>
__device__ void cudaReadBytes(void *dst, void const *src);

template <> __device__ void cudaReadBytes<8>(void *dst, void const *src) {
  reinterpret_cast<float2 *>(dst)[0] = reinterpret_cast<float2 const *>(src)[0];
}

template <> __device__ void cudaReadBytes<16>(void *dst, void const *src) {
  reinterpret_cast<float4 *>(dst)[0] = reinterpret_cast<float4 const *>(src)[0];
}

template <typename T_OUT, typename T_IN, size_t TPB, size_t VPT>
__global__ void cudaCast(T_OUT *dst, T_IN const *const src, const size_t size) {
  __shared__ T_IN srcTmps[TPB * VPT];
  __shared__ T_OUT dstTmps[TPB * VPT];
  for (size_t tid = (threadIdx.x + blockIdx.x * blockDim.x) * VPT; tid < size;
       tid += blockDim.x * gridDim.x * VPT) {
    cudaReadBytes<VPT * sizeof(T_IN)>(
        reinterpret_cast<void *>(srcTmps + threadIdx.x * VPT),
        reinterpret_cast<void const *>(src + tid));
    __syncwarp();
    int sharedid = threadIdx.x / 32 * (32 * VPT) + threadIdx.x % 32;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
      dstTmps[sharedid + i * 32] = (T_OUT)(srcTmps[sharedid + i * 32]);
    }
    __syncwarp();
    cudaReadBytes<VPT * sizeof(T_OUT)>(
        reinterpret_cast<void *>(dst + tid),
        reinterpret_cast<void const *>(dstTmps + threadIdx.x * VPT));
  }
}

template <size_t TPB, size_t VPT>
__global__ void cudaCastFloat2Half(half *dst, float const *const src,
                                   const size_t size) {
  __shared__ float2 srcTmps[TPB * VPT / 2];
  __shared__ half2 dstTmps[TPB * VPT / 2];
  for (size_t tid = (threadIdx.x + blockIdx.x * blockDim.x) * VPT; tid < size;
       tid += blockDim.x * gridDim.x * VPT) {
    cudaReadBytes<VPT * sizeof(float)>(
        reinterpret_cast<void *>(srcTmps + threadIdx.x * VPT / 2),
        reinterpret_cast<void const *>(src + tid));
    __syncwarp();
    int sharedid = threadIdx.x / 32 * (32 * VPT / 2) + threadIdx.x % 32;
#pragma unroll
    for (int i = 0; i < VPT / 2; ++i) {
      dstTmps[sharedid + i * 32] =
          __float22half2_rn(srcTmps[sharedid + i * 32]);
    }
    __syncwarp();
    cudaReadBytes<VPT * sizeof(half)>(
        reinterpret_cast<void *>(dst + tid),
        reinterpret_cast<void const *>(dstTmps + threadIdx.x * VPT / 2));
  }
}

template <size_t TPB, size_t VPT>
__global__ void cudaCastHalf2Float(float *dst, half const *const src,
                                   const size_t size) {
  __shared__ half2 srcTmps[TPB * VPT / 2];
  __shared__ float2 dstTmps[TPB * VPT / 2];
  for (size_t tid = (threadIdx.x + blockIdx.x * blockDim.x) * VPT; tid < size;
       tid += blockDim.x * gridDim.x * VPT) {
    cudaReadBytes<VPT * sizeof(half)>(
        reinterpret_cast<void *>(srcTmps + threadIdx.x * VPT / 2),
        reinterpret_cast<void const *>(src + tid));
    __syncwarp();
    int sharedid = threadIdx.x / 32 * (32 * VPT / 2) + threadIdx.x % 32;
#pragma unroll
    for (int i = 0; i < VPT / 2; ++i) {
      dstTmps[sharedid + i * 32] = __half22float2(srcTmps[sharedid + i * 32]);
    }
    __syncwarp();
    cudaReadBytes<VPT * sizeof(float)>(
        reinterpret_cast<void *>(dst + tid),
        reinterpret_cast<void const *>(dstTmps + threadIdx.x * VPT / 2));
  }
}

template <typename T_OUT, typename T_IN>
MLIR_GPU_OPS_EXPORT void invokeCudaCast(T_OUT *dst, T_IN const *const src,
                                        const size_t size,
                                        cudaStream_t stream) {
  cudaCast<T_OUT, T_IN, 256, 4><<<256, 256, 0, stream>>>(dst, src, size);
};

template <>
MLIR_GPU_OPS_EXPORT void
invokeCudaCast<half, float>(half *dst, float const *const src,
                            const size_t size, cudaStream_t stream) {
  cudaCastFloat2Half<256, 4><<<256, 256, 0, stream>>>(dst, src, size);
}

template <>
MLIR_GPU_OPS_EXPORT void
invokeCudaCast<float, half>(float *dst, half const *const src,
                            const size_t size, cudaStream_t stream) {
  cudaCastHalf2Float<256, 4><<<256, 256, 0, stream>>>(dst, src, size);
}