#include <cstdint>

#include "dnnl.hpp"

#ifdef _WIN32
#define MLIR_CPU_OPS_EXPORT __declspec(dllexport)
#else
#define MLIR_CPU_OPS_EXPORT
#endif // _WIN32

void cpuOpsInit(bool fastMathFlag = false);
dnnl::engine &getOpsEngine();
dnnl::stream &getOpsStream();
bool getOpsFastFlag();

void cpuLLMOpsInit();
void cpyLLMOpsDeinit();

extern "C" {
MLIR_CPU_OPS_EXPORT void mcpuMatmul(float *input, float *weight, float *bias,
                                    float *output, int64_t M, int64_t N,
                                    int64_t K);
MLIR_CPU_OPS_EXPORT void
mcpuConv2d(float *input, float *weight, float *bias, float *output,
           float *postAdd, int64_t N, int64_t IC, int64_t H, int64_t W,
           int64_t OC, int64_t KH, int64_t KW, int64_t OH, int64_t OW,
           int64_t PHL, int64_t PWL, int64_t PHR, int64_t PWR, int64_t SH,
           int64_t SW, int64_t DH, int64_t DW, bool hasPostAdd,
           bool hasContainRelu);

MLIR_CPU_OPS_EXPORT void mcpuPool2d(float *input, float *output, int64_t N,
                                    int64_t C, int64_t H, int64_t W, int64_t OH,
                                    int64_t OW, int64_t KH, int64_t KW,
                                    int64_t PHL, int64_t PWL, int64_t PHR,
                                    int64_t PWR, int64_t SH, int64_t SW,
                                    int64_t method /* 0 - max, 1 - avg */);

MLIR_CPU_OPS_EXPORT void mcpuLLMDecodingContextLayer(
    float *input, float *qkv, float *gemm_weight, float *gemm_bias,
    float *ffn1_weight, float *ffn1_bias, float *ffn2_weight, float *ffn2_bias,
    float *output, int64_t batch, int64_t seq_len, int64_t d_model,
    int64_t feed_forward_dim, int64_t head_num, bool norm_first, bool is_casual,
    bool is_relu);
}