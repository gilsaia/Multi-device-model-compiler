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
}