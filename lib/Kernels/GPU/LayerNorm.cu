#include "multi-device-model-compiler/Kernels/GPU/CudaReduceKernelUtils.cuh"
#include "multi-device-model-compiler/Kernels/GPU/CudaTypeUtils.cuh"
#include "multi-device-model-compiler/Kernels/GPU/Ops.h"

#include "cuda_fp16.h"

#include <assert.h>

// * Note that typename T is half2 or bfloat2 type
template <typename T, bool IS_OUTPUT, bool IS_BIAS, int RESIDUAL_NUM,
          bool IS_BETA, int UNROLL_FACTOR>
__global__ void generalAddBiasResidualLayerNormOpt(
    T *normed_output, T *output, const T *__restrict input,
    const T *__restrict bias, const T *__restrict residual1,
    const T *__restrict residual2, const T *__restrict gamma,
    const T *__restrict beta, const float layernorm_eps, int m, int n,
    const float *scale_inter, const float *scale_out, const float *scale,
    float *dynamic_scale, const int int8_mode) {
  extern __shared__ __align__(
      sizeof(float)) char _shmem[]; // Align on largest type
  T *shmem = reinterpret_cast<T *>(_shmem);

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;

  using Int8_Packed_T = typename packed_as<int8_t, num_elems<T>::value>::type;
  using Int32_Packed_T = typename packed_as<int32_t, num_elems<T>::value>::type;
  using Float_Packed_T = typename packed_as<float, num_elems<T>::value>::type;
  using Scalar_T = typename packed_as<T, 1>::type;

  const bool scale_input = int8_mode == 2 && scale_inter != nullptr;
  const bool dynamic_scaling = dynamic_scale != nullptr;

  T local_sum = cuda_cast<T>(0.0f);

  const Float_Packed_T scale_from_int = cuda_cast<Float_Packed_T>(
      scale_input ? (*scale_inter) * (*scale_out) : 0.0f);
  const Float_Packed_T scale_to_int =
      cuda_cast<Float_Packed_T>(int8_mode == 2 ? *scale : 0.0f);

#pragma unroll
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int index = blockIdx.x * n + i;
    T val = cuda_cast<T>(0.0f);

    if (IS_BIAS) {
      val = hadd2(val, ldg(&bias[i]));
    }
    if (RESIDUAL_NUM == 1) {
      val = hadd2(val, ldg(&residual1[index]));
    } else if (RESIDUAL_NUM == 2) {
      val = hadd2(hadd2(val, ldg(&residual1[index])), ldg(&residual2[index]));
    }

    if (IS_OUTPUT) {
      T in_val;
      if (scale_input) {
        in_val = cuda_cast<T>(
            cuda_cast<Float_Packed_T>(
                reinterpret_cast<const Int32_Packed_T *>(input)[index]) *
            scale_from_int);
      } else {
        in_val = input[index];
      }
      val = hadd2(val, in_val);
    }
    shmem[i] = val;
    output[index] = val;
    local_sum = hadd2(local_sum, val);
  }

  mean = blockReduceSum((float)(local_sum.x + local_sum.y));

  if (threadIdx.x == 0) {
    s_mean = mean / n / 2;
  }
  __syncthreads();

  float local_var_sum = 0.0f;
#pragma unroll UNROLL_FACTOR
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    T val = input[blockIdx.x * n + i];
    float diff_1 = (float)(val.x) - s_mean;
    float diff_2 = (float)(val.y) - s_mean;
    local_var_sum += (diff_1 * diff_1 + diff_2 * diff_2);
  }
  variance = blockReduceSum(local_var_sum);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / n / 2 + layernorm_eps);
  }
  __syncthreads();

  T mean_2 = cuda_cast<T>(s_mean);
  T var_2 = cuda_cast<T>(s_variance);

  Scalar_T abs_max = 1e-6f;

#pragma unroll UNROLL_FACTOR
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int index = blockIdx.x * n + i;
    T val = hmul2(hsub2(shmem[i], mean_2), var_2, ldg(&gamma[i]));
    if (IS_BETA) {
      val = hadd2(val, ldg(&beta[i]));
    }

    if (dynamic_scaling) {
      abs_max = cuda_max(cuda_max<Scalar_T>(cuda_abs(val)), abs_max);
      shmem[i] = val;
    } else if (int8_mode == 2) {
      reinterpret_cast<Int8_Packed_T *>(normed_output)[index] =
          cuda_cast<Int8_Packed_T>(cuda_cast<Float_Packed_T>(val) *
                                   scale_to_int);
    } else {
      normed_output[index] = val;
    }
  }

  if (dynamic_scaling) {
    float abs_max_f = blockAllReduceMax(cuda_cast<float>(abs_max));
    const float dynamic_per_token_scale = 127. / abs_max_f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      const int index = blockIdx.x * n + i;
      reinterpret_cast<Int8_Packed_T *>(normed_output)[index] =
          cuda_cast<Int8_Packed_T>(
              cuda_cast<Float_Packed_T>(shmem[i]) *
              cuda_cast<Float_Packed_T>(dynamic_per_token_scale));
    }
    if (threadIdx.x == 0) {
      dynamic_scale[blockIdx.x] = (*scale * abs_max_f) / 127.f;
    }
  }
}

// * Note that typename T is half2 or bfloat2 type
template <typename T, bool IS_OUTPUT, bool IS_BIAS, int RESIDUAL_NUM,
          bool IS_BETA, int UNROLL_FACTOR>
__global__ void generalAddBiasResidualLayerNormOpt2(
    T *normed_output, T *output, const T *__restrict input,
    const T *__restrict bias, const T *__restrict residual1,
    const T *__restrict residual2, const T *__restrict gamma,
    const T *__restrict beta, const float layernorm_eps, int m, int n,
    const float *scale_inter, const float *scale_out, const float *scale,
    float *dynamic_scale, const int int8_mode) {
  extern __shared__ __align__(sizeof(float)) char _shmem[];
  T *shmem = reinterpret_cast<T *>(_shmem);

  __shared__ float s_mean;
  __shared__ float s_variance;
  float x_sum = 0.0f;
  float x2_sum = 0.0f;
  const int b_offset = blockIdx.x * n;

  using T1 = typename TypeConverter<T>::Type;
  using Int8_Packed_T = typename packed_as<int8_t, num_elems<T>::value>::type;
  using Int32_Packed_T = typename packed_as<int32_t, num_elems<T>::value>::type;
  using Float_Packed_T = typename packed_as<float, num_elems<T>::value>::type;
  using Scalar_T = typename packed_as<T, 1>::type;

  const bool scale_input = int8_mode == 2 && scale_inter != nullptr;
  const Float_Packed_T scale_vec_in = cuda_cast<Float_Packed_T>(
      scale_input ? (*scale_inter) * (*scale_out) : 0.0f);
  const Float_Packed_T scale_vec =
      cuda_cast<Float_Packed_T>(int8_mode == 2 ? *scale : 0.0f);
  const bool dynamic_scaling = dynamic_scale != nullptr;

#pragma unroll UNROLL_FACTOR
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int index = b_offset + i;
    float val_1 = 0.0f;
    float val_2 = 0.0f;
    T tmp;

    if (IS_BIAS) {
      tmp = ldg(&bias[i]);
      val_1 += static_cast<float>(tmp.x);
      val_2 += static_cast<float>(tmp.y);
    }
    if (RESIDUAL_NUM == 1) {
      tmp = ldg(&residual1[index]);
      val_1 += static_cast<float>(tmp.x);
      val_2 += static_cast<float>(tmp.y);
    } else if (RESIDUAL_NUM == 2) {
      tmp = ldg(&residual1[index]);
      T tmp2 = ldg(&residual2[index]);
      val_1 += (static_cast<float>(tmp.x) + static_cast<float>(tmp2.x));
      val_2 += (static_cast<float>(tmp.y) + static_cast<float>(tmp2.y));
    }

    if (IS_OUTPUT) {
      if (scale_input) {
        tmp = cuda_cast<T>(
            cuda_cast<Float_Packed_T>(
                reinterpret_cast<const Int32_Packed_T *>(input)[index]) *
            scale_vec_in);
      } else {
        tmp = ldg(&input[index]);
      }
      val_1 += static_cast<float>(tmp.x);
      val_2 += static_cast<float>(tmp.y);
    }
    tmp.x = cuda_cast<T1>(val_1);
    tmp.y = cuda_cast<T1>(val_2);
    shmem[i] = tmp;
    output[index] = tmp;
    x_sum += val_1 + val_2;
    x2_sum += val_1 * val_1 + val_2 * val_2;
  }
  float sums[2];
  sums[0] = x_sum;
  sums[1] = x2_sum;
  blockReduceSumV2<float, 2>(sums);

  if (threadIdx.x == 0) {
    s_mean = sums[0] / n / 2;
    s_variance = rsqrtf(sums[1] / n / 2 - s_mean * s_mean + layernorm_eps);
  }
  __syncthreads();

  T mean_2 = cuda_cast<T>(s_mean);
  T var_2 = cuda_cast<T>(s_variance);

  Scalar_T abs_max = 1e-6f;

#pragma unroll UNROLL_FACTOR
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int index = blockIdx.x * n + i;
    T val = hmul2(hsub2(shmem[i], mean_2), var_2, ldg(&gamma[i]));
    if (IS_BETA) {
      val = hadd2(val, ldg(&beta[i]));
    }

    if (dynamic_scaling) {
      abs_max = cuda_max(cuda_max<Scalar_T>(cuda_abs(val)), abs_max);
      shmem[i] = val;
    } else if (int8_mode == 2) {
      reinterpret_cast<Int8_Packed_T *>(normed_output)[index] =
          cuda_cast<Int8_Packed_T>(cuda_cast<Float_Packed_T>(val) * scale_vec);
    } else {
      normed_output[index] = val;
    }
  }

  if (dynamic_scaling) {
    float abs_max_f = blockAllReduceMax(cuda_cast<float>(abs_max));
    const float dynamic_per_token_scale = 127. / abs_max_f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      const int index = blockIdx.x * n + i;
      reinterpret_cast<Int8_Packed_T *>(normed_output)[index] =
          cuda_cast<Int8_Packed_T>(
              cuda_cast<Float_Packed_T>(shmem[i]) *
              cuda_cast<Float_Packed_T>(dynamic_per_token_scale));
    }
    if (threadIdx.x == 0) {
      dynamic_scale[blockIdx.x] = (*scale * abs_max_f) / 127.f;
    }
  }
}

template <typename T, bool IS_OUTPUT, bool IS_BIAS, int UNROLL_FACTOR,
          int RESIDUAL_NUM>
void dispatch_generalAddBiasResidualLayerNormOpt_opt_version(
    T *norm_output, T *output, const T *input, const T *bias,
    const T *residual1, const T *residual2, const T *gamma, const T *beta,
    float layernorm_eps, int m, int half_n, const float *scale_inter,
    const float *scale_out, const float *scale, float *dynamic_scale,
    int int8_mode, dim3 grid, dim3 block, cudaStream_t stream,
    int opt_version) {
  size_t maxbytes = half_n * sizeof(T);
  if (opt_version == 1) {
    if (maxbytes >= (48 << 10)) {
      checkCudaStatus(cudaFuncSetAttribute(
          generalAddBiasResidualLayerNormOpt<T, IS_OUTPUT, IS_BIAS,
                                             RESIDUAL_NUM, true, UNROLL_FACTOR>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
    }
    generalAddBiasResidualLayerNormOpt<T, IS_OUTPUT, IS_BIAS, RESIDUAL_NUM,
                                       true, UNROLL_FACTOR>
        <<<grid, block, maxbytes, stream>>>(
            norm_output, output, input, bias, residual1, residual2, gamma, beta,
            layernorm_eps, m, half_n, scale_inter, scale_out, scale,
            dynamic_scale, int8_mode);
  } else if (opt_version == 2) {
    if (maxbytes >= (48 << 10)) {
      checkCudaStatus(cudaFuncSetAttribute(
          generalAddBiasResidualLayerNormOpt2<
              T, IS_OUTPUT, IS_BIAS, RESIDUAL_NUM, true, UNROLL_FACTOR>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
    }
    generalAddBiasResidualLayerNormOpt2<T, IS_OUTPUT, IS_BIAS, RESIDUAL_NUM,
                                        true, UNROLL_FACTOR>
        <<<grid, block, maxbytes, stream>>>(
            norm_output, output, input, bias, residual1, residual2, gamma, beta,
            layernorm_eps, m, half_n, scale_inter, scale_out, scale,
            dynamic_scale, int8_mode);
  } else {
    assert(false && "opt_num must be 1 or 2");
  }
}

template <typename T, bool IS_BIAS, int UNROLL_FACTOR, int RESIDUAL_NUM>
void dispatch_generalAddBiasResidualLayerNormOpt_is_output(
    T *norm_output, T *output, const T *input, const T *bias,
    const T *residual1, const T *residual2, const T *gamma, const T *beta,
    float layernorm_eps, int m, int half_n, const float *scale_inter,
    const float *scale_out, const float *scale, float *dynamic_scale,
    int int8_mode, dim3 grid, dim3 block, cudaStream_t stream, int opt_version,
    bool is_output) {
  if (is_output) {
    dispatch_generalAddBiasResidualLayerNormOpt_opt_version<
        T, true, IS_BIAS, UNROLL_FACTOR, RESIDUAL_NUM>(
        norm_output, output, input, bias, residual1, residual2, gamma, beta,
        layernorm_eps, m, half_n, scale_inter, scale_out, scale, dynamic_scale,
        int8_mode, grid, block, stream, opt_version);
  } else {
    dispatch_generalAddBiasResidualLayerNormOpt_opt_version<
        T, false, IS_BIAS, UNROLL_FACTOR, RESIDUAL_NUM>(
        norm_output, output, input, bias, residual1, residual2, gamma, beta,
        layernorm_eps, m, half_n, scale_inter, scale_out, scale, dynamic_scale,
        int8_mode, grid, block, stream, opt_version);
  }
}

template <typename T, int UNROLL_FACTOR, int RESIDUAL_NUM>
void dispatch_generalAddBiasResidualLayerNormOpt_bias(
    T *norm_output, T *output, const T *input, const T *bias,
    const T *residual1, const T *residual2, const T *gamma, const T *beta,
    float layernorm_eps, int m, int half_n, const float *scale_inter,
    const float *scale_out, const float *scale, float *dynamic_scale,
    int int8_mode, dim3 grid, dim3 block, cudaStream_t stream, int opt_version,
    bool is_output) {
  if (bias != nullptr) {
    dispatch_generalAddBiasResidualLayerNormOpt_is_output<
        T, true, UNROLL_FACTOR, RESIDUAL_NUM>(
        norm_output, output, input, bias, residual1, residual2, gamma, beta,
        layernorm_eps, m, half_n, scale_inter, scale_out, scale, dynamic_scale,
        int8_mode, grid, block, stream, opt_version, is_output);
  } else {
    dispatch_generalAddBiasResidualLayerNormOpt_is_output<
        T, false, UNROLL_FACTOR, RESIDUAL_NUM>(
        norm_output, output, input, bias, residual1, residual2, gamma, beta,
        layernorm_eps, m, half_n, scale_inter, scale_out, scale, dynamic_scale,
        int8_mode, grid, block, stream, opt_version, is_output);
  }
}

template <typename T, int UNROLL_FACTOR>
void dispatch_generalAddBiasResidualLayerNormOpt_residual_num(
    T *norm_output, T *output, const T *input, const T *bias,
    const T *residual1, const T *residual2, const T *gamma, const T *beta,
    float layernorm_eps, int m, int half_n, const float *scale_inter,
    const float *scale_out, const float *scale, float *dynamic_scale,
    int int8_mode, dim3 grid, dim3 block, cudaStream_t stream, int opt_version,
    bool is_output, int residual_num) {
  if (residual_num == 1) {
    dispatch_generalAddBiasResidualLayerNormOpt_bias<T, UNROLL_FACTOR, 1>(
        norm_output, output, input, bias, residual1, residual2, gamma, beta,
        layernorm_eps, m, half_n, scale_inter, scale_out, scale, dynamic_scale,
        int8_mode, grid, block, stream, opt_version, is_output);
  } else if (residual_num == 2) {
    dispatch_generalAddBiasResidualLayerNormOpt_bias<T, UNROLL_FACTOR, 2>(
        norm_output, output, input, bias, residual1, residual2, gamma, beta,
        layernorm_eps, m, half_n, scale_inter, scale_out, scale, dynamic_scale,
        int8_mode, grid, block, stream, opt_version, is_output);
  } else {
    assert(false && "residual_num must be 1 or 2");
  }
}

template <typename T>
void dispatch_generalAddBiasResidualLayerNormOpt_unroll_factor(
    T *norm_output, T *output, const T *input, const T *bias,
    const T *residual1, const T *residual2, const T *gamma, const T *beta,
    float layernorm_eps, int m, int half_n, const float *scale_inter,
    const float *scale_out, const float *scale, float *dynamic_scale,
    int int8_mode, dim3 grid, dim3 block, cudaStream_t stream, int opt_version,
    bool is_output, int residual_num, int unroll_factor) {
  switch (unroll_factor) {
  case 1:
    dispatch_generalAddBiasResidualLayerNormOpt_residual_num<T, 1>(
        norm_output, output, input, bias, residual1, residual2, gamma, beta,
        layernorm_eps, m, half_n, scale_inter, scale_out, scale, dynamic_scale,
        int8_mode, grid, block, stream, opt_version, is_output, residual_num);
    break;
  case 2:
    dispatch_generalAddBiasResidualLayerNormOpt_residual_num<T, 2>(
        norm_output, output, input, bias, residual1, residual2, gamma, beta,
        layernorm_eps, m, half_n, scale_inter, scale_out, scale, dynamic_scale,
        int8_mode, grid, block, stream, opt_version, is_output, residual_num);
    break;
  case 4:
    dispatch_generalAddBiasResidualLayerNormOpt_residual_num<T, 4>(
        norm_output, output, input, bias, residual1, residual2, gamma, beta,
        layernorm_eps, m, half_n, scale_inter, scale_out, scale, dynamic_scale,
        int8_mode, grid, block, stream, opt_version, is_output, residual_num);
    break;
  case 8:
    dispatch_generalAddBiasResidualLayerNormOpt_residual_num<T, 8>(
        norm_output, output, input, bias, residual1, residual2, gamma, beta,
        layernorm_eps, m, half_n, scale_inter, scale_out, scale, dynamic_scale,
        int8_mode, grid, block, stream, opt_version, is_output, residual_num);
    break;
  default:
    assert(false && "unroll_factor must be 1, 2, 4 or 8");
  }
}

template <typename T>
MLIR_GPU_OPS_EXPORT void
mgpuLayerNorm(const T *input, const T *gamma, const T *beta, T *output,
              const float layernorm_eps, const int m, const int n, float *scale,
              cudaStream_t stream) {
  // only consider half
  int half_n = n / 2;
  int half_n_32 = (half_n + 31) / 32 * 32;
  dim3 grid(m);
  dim3 block(min(half_n_32, 512));
  int rolls_per_thread = half_n / block.x;
  int unroll_factor = 8;
  while (unroll_factor > rolls_per_thread && unroll_factor > 1) {
    unroll_factor /= 2;
  }

  dispatch_generalAddBiasResidualLayerNormOpt_unroll_factor<half2>(
      (half2 *)output, (half2 *)output, (const half2 *)output, nullptr,
      (const half2 *)input, nullptr, (const half2 *)gamma, (const half2 *)beta,
      layernorm_eps, m, half_n, nullptr, nullptr, scale, nullptr, 0, grid,
      block, stream, 2, false, 1, unroll_factor);
}

template MLIR_GPU_OPS_EXPORT void
mgpuLayerNorm<half>(const half *input, const half *gamma, const half *beta,
                    half *output, const float layernorm_eps, const int m,
                    const int n, float *scale, cudaStream_t stream);

template <typename T>
MLIR_GPU_OPS_EXPORT void
mgpuAddResidualPreLayerNorm(const T *input, const T *residual, const T *gamma,
                            const T *beta, T *output, const float layernorm_eps,
                            int m, int n, float *scale, cudaStream_t stream) {
  // only consider half
  int half_n = n / 2;
  int half_n_32 = (half_n + 31) / 32 * 32;
  dim3 grid(m);
  dim3 block(min(half_n_32, 512));
  int rolls_per_thread = half_n / block.x;
  int unroll_factor = 8;
  while (unroll_factor > rolls_per_thread && unroll_factor > 1) {
    unroll_factor /= 2;
  }

  dispatch_generalAddBiasResidualLayerNormOpt_unroll_factor<half2>(
      (half2 *)output, (half2 *)output, (const half2 *)output, nullptr,
      (const half2 *)input, (const half2 *)residual, (const half2 *)gamma,
      (const half2 *)beta, layernorm_eps, m, half_n, nullptr, nullptr, scale,
      nullptr, 0, grid, block, stream, 2, false, 2, unroll_factor);
}

template MLIR_GPU_OPS_EXPORT void mgpuAddResidualPreLayerNorm<half>(
    const half *input, const half *residual, const half *gamma,
    const half *beta, half *output, const float layernorm_eps, const int m,
    const int n, float *scale, cudaStream_t stream);