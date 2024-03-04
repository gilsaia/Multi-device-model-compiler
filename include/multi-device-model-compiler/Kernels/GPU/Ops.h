#ifndef MULTI_DEVICE_MODEL_COMPILER_KERNELS_GPU_OPS_H_
#define MULTI_DEVICE_MODEL_COMPILER_KERNELS_GPU_OPS_H_

#include <cstdint>

#include "cuda.h"
#include "cuda_runtime.h"

#ifdef _WIN32
#define MLIR_GPU_OPS_EXPORT __declspec(dllexport)
#else
#define MLIR_GPU_OPS_EXPORT
#endif // _WIN32

void gpuOpsInit();
void gpuOpsDeinit();
void gpuLLMOpsInit();
void gpuLLMOpsDeinit();

void checkCudaStatus(cudaError_t status);

extern "C" {
MLIR_GPU_OPS_EXPORT void mgpuMatmul(float *input, float *weight, float *bias,
                                    float *output, int64_t M, int64_t N,
                                    int64_t K, cudaStream_t stream);

MLIR_GPU_OPS_EXPORT void
mgpuConv2d(float *input, float *weight, float *bias, float *output,
           float *postAdd, int64_t N, int64_t IC, int64_t H, int64_t W,
           int64_t OC, int64_t KH, int64_t KW, int64_t OH, int64_t OW,
           int64_t PHL, int64_t PWL, int64_t PHR, int64_t PWR, int64_t SH,
           int64_t SW, int64_t DH, int64_t DW, bool hasPostAdd,
           bool hasContainRelu, cudaStream_t stream);

MLIR_GPU_OPS_EXPORT void mgpuPool2d(float *input, float *output, int64_t N,
                                    int64_t C, int64_t H, int64_t W, int64_t OH,
                                    int64_t OW, int64_t KH, int64_t KW,
                                    int64_t PHL, int64_t PWL, int64_t PHR,
                                    int64_t PWR, int64_t SH, int64_t SW,
                                    int64_t method /* 0 - max, 1 - avg */,
                                    cudaStream_t stream);

MLIR_GPU_OPS_EXPORT void mgpuLLMDecodingContextLayer(
    float *input, float *qkv, float *gemm_weight, float *gemm_bias,
    float *ffn1_weight, float *ffn1_bias, float *ffn2_weight, float *ffn2_bias,
    float *output, int64_t batch, int64_t seq_len, int64_t d_model,
    int64_t feed_forward_dim, int64_t head_num, bool norm_first, bool is_casual,
    bool is_relu, cudaStream_t stream);
}

template <typename T>
MLIR_GPU_OPS_EXPORT void mgpuMatmulEx(T *input, T *weight, T *bias, T *output,
                                      T *residual, int64_t M, int64_t N,
                                      int64_t K, bool hasBias, bool hasRelu,
                                      bool hasResidual, cudaStream_t stream);

template <typename T>
MLIR_GPU_OPS_EXPORT void
mgpuLayerNorm(const T *input, const T *gamma, const T *beta, T *output,
              const float layernorm_eps, const int m, const int n, float *scale,
              cudaStream_t stream);

template <typename T>
MLIR_GPU_OPS_EXPORT void
mgpuAddResidualPreLayerNorm(const T *input, const T *residual, const T *gamma,
                            const T *beta, T *output, T *normed_output,
                            const float layernorm_eps, int m, int n,
                            float *scale, cudaStream_t stream);

#endif