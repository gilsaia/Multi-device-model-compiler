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
}