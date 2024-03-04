#include "multi-device-model-compiler/Kernels/GPU/CudaTypeUtils.cuh"
#include "multi-device-model-compiler/Kernels/GPU/Ops.h"

#include "cuda_fp16.h"

template <typename T>
__global__ void add_fusedQKV_bias_transpose_kernel(
    T *q_buf, T *k_buf, T *v_buf, T *QKV, const T *__restrict qkv_bias,
    const int *padding_offset, const int batch_size, const int seq_len,
    const int token_num, const int head_num, const int size_per_head,
    const float *scale, const int int8_mode) {
  // QKV: [token_num, 3, n]
  // qkv_bias: [3, n]
  // q_buf, k_buf, v_buf: [batch, head_num, seq_len, size_per_head]

  T *qkv_ptr[3] = {q_buf, k_buf, v_buf};
  const int n = head_num * size_per_head;
  for (int index = blockDim.x * blockIdx.x + threadIdx.x;
       index < token_num * 3 * n; index += gridDim.x * blockDim.x) {
    const int bias_id = index % (3 * n);

    const int token_idx = index / (3 * n);
    const int token_padded_idx =
        token_idx + (padding_offset == nullptr ? 0 : padding_offset[token_idx]);
    const int target_batch_id = token_padded_idx / seq_len;
    const int seq_id = token_padded_idx % seq_len;

    const int qkv_id = (index % (3 * n)) / n;
    const int head_id = (index % n) / size_per_head;
    const int size_id = index % size_per_head;

    T val;
    if (int8_mode == 2) {
      val = cuda_cast<T>(
          cuda_cast<float>(reinterpret_cast<const int8_t *>(QKV)[index]) *
          scale[qkv_id]);
    } else {
      val = ldg(&QKV[index]);
    }
    val = val + ldg(&qkv_bias[bias_id]);

    if (int8_mode == 2) {
      // TODO(mseznec): add support for int8 BMM with FusedAtt
    } else {
      QKV[index] = val;
    }

    qkv_ptr[qkv_id][target_batch_id * head_num * seq_len * size_per_head +
                    head_id * seq_len * size_per_head + seq_id * size_per_head +
                    size_id] = val;
  }
}

template <typename T>
void invokeAddFusedQKVBiasTranspose(
    T *q_buf, T *k_buf, T *v_buf, T *QKV, const T *qkv_bias,
    const int *padding_offset, const int batch_size, const int seq_len,
    const int token_num, const int head_num, const int size_per_head,
    const int rotary_embedding_dim, const int neox_rotary_style,
    const float *scale, const int int8_mode, cudaStream_t stream) {
  // [bs, seq_len, 3, head, Dh]
  const int m = token_num;
  const int n = head_num * size_per_head;
  dim3 block(384);
  dim3 grid((int)(ceil(1.0 * m * n / 384)));
  add_fusedQKV_bias_transpose_kernel<<<grid, block, 0, stream>>>(
      q_buf, k_buf, v_buf, QKV, qkv_bias, padding_offset, batch_size, seq_len,
      token_num, head_num, size_per_head, scale, int8_mode);
}

template <typename T>
MLIR_GPU_OPS_EXPORT void mgpuAddFusedQKVBiasTranspose(
    T *q_buf, T *k_buf, T *v_buf, T *QKV, const T *qkv_bias,
    const int batch_size, const int seq_len, const int token_num,
    const int head_num, const int size_per_head, cudaStream_t stream) {
  int padding_offset = 0;
  invokeAddFusedQKVBiasTranspose<T>(
      q_buf, k_buf, v_buf, QKV, qkv_bias, &padding_offset, batch_size, seq_len,
      token_num, head_num, size_per_head, 0, false, nullptr, 0, stream);
}

template MLIR_GPU_OPS_EXPORT void mgpuAddFusedQKVBiasTranspose<half>(
    half *q_buf, half *k_buf, half *v_buf, half *QKV, const half *qkv_bias,
    const int batch_size, const int seq_len, const int token_num,
    const int head_num, const int size_per_head, cudaStream_t stream);