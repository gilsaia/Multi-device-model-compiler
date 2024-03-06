#include "multi-device-model-compiler/Kernels/GPU/CudaTypeUtils.cuh"
#include "multi-device-model-compiler/Kernels/GPU/Ops.h"
#include "multi-device-model-compiler/Kernels/GPU/Utils.h"
#include "multi-device-model-compiler/Runtime/CUDA/CudaRuntimeWrappers.h"
#include "qkvToContext.h"

#include "cuda_fp16.h"

cudaStream_t LLMCopyStream;
cudaEvent_t LLMCopyEvent;
int sm;
std::unique_ptr<fastertransformer::MHARunner> dispatcher_fp16;

void gpuLLMOpsInit() {
  cudaStreamCreate(&LLMCopyStream);
  cudaEventCreate(&LLMCopyEvent);
  sm = getSMVersion();
  dispatcher_fp16.reset(
      new fastertransformer::FusedMHARunnerFP16v2(8, 128, sm, 1.0f));
}
void gpuLLMOpsDeinit() {
  cudaStreamDestroy(LLMCopyStream);
  cudaEventDestroy(LLMCopyEvent);
}

extern "C" MLIR_GPU_OPS_EXPORT void mgpuLLMDecodingContextLayer(
    float *input, float *qkv, float *gemm_weight, float *gemm_bias,
    float *ffn1_weight, float *ffn1_bias, float *ffn2_weight, float *ffn2_bias,
    float *output, int64_t batch, int64_t seq_len, int64_t d_model,
    int64_t feed_forward_dim, int64_t head_num, bool norm_first, bool is_casual,
    bool is_relu, cudaStream_t stream) {
  int64_t d_head = d_model / head_num;

  // transform input to half
  half *input_h = reinterpret_cast<half *>(mgpuMemAllocAsync(
      batch * seq_len * d_model * sizeof(half), LLMCopyStream));
  invokeCudaCast(input_h, input, batch * seq_len * d_model, LLMCopyStream);
  // alloc layernorm output
  half *input_layernorm_h = reinterpret_cast<half *>(mgpuMemAllocAsync(
      batch * seq_len * d_model * sizeof(half), LLMCopyStream));
  half *layernorm_gamma = reinterpret_cast<half *>(
      mgpuMemAllocAsync(d_model * sizeof(half), LLMCopyStream));
  half *layernorm_beta = reinterpret_cast<half *>(
      mgpuMemAllocAsync(d_model * sizeof(half), LLMCopyStream));
  cudaMemsetAsync(layernorm_beta, 0, d_model * sizeof(half), LLMCopyStream);
  deviceFill(layernorm_gamma, d_model, __float2half(1), LLMCopyStream);
  cudaEventRecord(LLMCopyEvent, LLMCopyStream);

  // perform layernorm on input
  cudaStreamWaitEvent(stream, LLMCopyEvent);
  mgpuLayerNorm(input_h, layernorm_gamma, layernorm_beta, input_layernorm_h,
                1e-5, batch * seq_len, d_model, nullptr, stream);

  // transform QKV to half
  half *qkv_h = reinterpret_cast<half *>(
      mgpuMemAllocAsync(d_model * d_model * 3 * sizeof(half), LLMCopyStream));
  invokeCudaCast(qkv_h, qkv, d_model * d_model * 3, LLMCopyStream);
  // alloc qkv gemm output
  half *input_qkv_h = reinterpret_cast<half *>(mgpuMemAllocAsync(
      batch * seq_len * d_model * 3 * sizeof(half), LLMCopyStream));

  int *cu_seqlens = reinterpret_cast<int *>(
      mgpuMemAllocAsync(batch * sizeof(int), LLMCopyStream));
  deviceFill<int>(cu_seqlens, batch, seq_len, LLMCopyStream);
  cudaEventRecord(LLMCopyEvent, LLMCopyStream);

  // perform qkv gemm
  // wait qkv transform
  cudaStreamWaitEvent(stream, LLMCopyEvent);
  mgpuMatmulEx<half>(input_layernorm_h, qkv_h, nullptr, input_qkv_h, nullptr,
                     seq_len * batch, d_model * 3, d_model, false, false, false,
                     stream);
  // split qkv output
  half *input_q_h = input_qkv_h,
       *input_k_h = input_qkv_h + seq_len * batch * d_model,
       *input_v_h = input_qkv_h + (seq_len * batch * d_model * 2);

  // perform self-attn call
  dispatcher_fp16->setup_causal_masked_fmha(seq_len, batch);
  dispatcher_fp16->run_causal_masked_fmha(input_qkv_h, cu_seqlens,
                                          input_layernorm_h, true, stream);

  // alloc proj gemm weight bias
  half *proj_gemm_weight_h = reinterpret_cast<half *>(
      mgpuMemAllocAsync(d_model * d_model * sizeof(half), LLMCopyStream));
  invokeCudaCast(proj_gemm_weight_h, gemm_weight, d_model * d_model,
                 LLMCopyStream);
  half *proj_gemm_bias_h = reinterpret_cast<half *>(
      mgpuMemAllocAsync(d_model * sizeof(half), LLMCopyStream));
  invokeCudaCast(proj_gemm_bias_h, gemm_bias, d_model, LLMCopyStream);
  cudaEventRecord(LLMCopyEvent, LLMCopyStream);

  // perform proj gemm
  cudaStreamWaitEvent(stream, LLMCopyEvent);
  mgpuMatmulEx<half>(input_layernorm_h, proj_gemm_weight_h, proj_gemm_bias_h,
                     input_qkv_h, nullptr, batch * seq_len, d_model, d_model,
                     true, false, false, stream);

  // alloc ffn1 weight bias output
  half *ffn1_weight_h = reinterpret_cast<half *>(mgpuMemAllocAsync(
      d_model * feed_forward_dim * sizeof(half), LLMCopyStream));
  invokeCudaCast(ffn1_weight_h, ffn1_weight, d_model * feed_forward_dim,
                 LLMCopyStream);
  half *ffn1_bias_h = reinterpret_cast<half *>(
      mgpuMemAllocAsync(feed_forward_dim * sizeof(half), LLMCopyStream));
  invokeCudaCast(ffn1_bias_h, ffn1_bias, feed_forward_dim, LLMCopyStream);
  half *ffn1_output_h = reinterpret_cast<half *>(mgpuMemAllocAsync(
      batch * seq_len * feed_forward_dim * sizeof(half), LLMCopyStream));
  cudaEventRecord(LLMCopyEvent, LLMCopyStream);

  // perform add&layernorm call
  mgpuAddResidualPreLayerNorm(input_qkv_h, input_h, layernorm_gamma,
                              layernorm_beta, input_k_h, input_layernorm_h,
                              1e-5, batch * seq_len, d_model, nullptr, stream);

  // perform ffn1
  // wait ffn1 transform
  cudaStreamWaitEvent(stream, LLMCopyEvent);
  mgpuMatmulEx<half>(input_layernorm_h, ffn1_weight_h, ffn1_bias_h,
                     ffn1_output_h, nullptr, batch * seq_len, feed_forward_dim,
                     d_model, true, true, false, stream);

  // alloc ffn2 weight bias output
  half *ffn2_weight_h = reinterpret_cast<half *>(mgpuMemAllocAsync(
      feed_forward_dim * d_model * sizeof(half), LLMCopyStream));
  invokeCudaCast(ffn2_weight_h, ffn2_weight, feed_forward_dim * d_model,
                 LLMCopyStream);
  half *ffn2_bias_h = reinterpret_cast<half *>(
      mgpuMemAllocAsync(d_model * sizeof(half), LLMCopyStream));
  invokeCudaCast(ffn2_bias_h, ffn2_bias, d_model, LLMCopyStream);
  cudaEventRecord(LLMCopyEvent, LLMCopyStream);

  // perform ffn2
  // wait ffn2 transform
  cudaStreamWaitEvent(stream, LLMCopyEvent);
  mgpuMatmulEx<half>(ffn1_output_h, ffn2_weight_h, ffn2_bias_h, input_h,
                     input_k_h, batch * seq_len, d_model, feed_forward_dim,
                     true, false, true, stream);

  // transform input to output
  invokeCudaCast(output, input_h, batch * seq_len * d_model, LLMCopyStream);
  // dealloc
  mgpuMemFreeAsync(input_h, LLMCopyStream);
  mgpuMemFreeAsync(input_layernorm_h, LLMCopyStream);
  mgpuMemFreeAsync(qkv_h, LLMCopyStream);
  mgpuMemFreeAsync(input_qkv_h, LLMCopyStream);
  mgpuMemFreeAsync(proj_gemm_weight_h, LLMCopyStream);
  mgpuMemFreeAsync(proj_gemm_bias_h, LLMCopyStream);
  mgpuMemFreeAsync(ffn1_weight_h, LLMCopyStream);
  mgpuMemFreeAsync(ffn1_bias_h, LLMCopyStream);
  mgpuMemFreeAsync(ffn1_output_h, LLMCopyStream);
  mgpuMemFreeAsync(ffn2_weight_h, LLMCopyStream);
  mgpuMemFreeAsync(ffn2_bias_h, LLMCopyStream);
  mgpuMemFreeAsync(cu_seqlens, LLMCopyStream);
  cudaEventRecord(LLMCopyEvent, LLMCopyStream);

  // add dependency
  cudaStreamWaitEvent(stream, LLMCopyEvent);
}