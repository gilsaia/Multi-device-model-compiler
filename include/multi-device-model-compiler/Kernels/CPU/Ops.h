#include <cstdint>

#ifdef _WIN32
#define MLIR_CPU_OPS_EXPORT __declspec(dllexport)
#else
#define MLIR_CPU_OPS_EXPORT
#endif // _WIN32

void cpuOpsInit();

extern "C" {
MLIR_CPU_OPS_EXPORT void mcpuMatmul(float *input, float *weight, float *bias,
                                    float *output, int64_t M, int64_t N,
                                    int64_t K);
}