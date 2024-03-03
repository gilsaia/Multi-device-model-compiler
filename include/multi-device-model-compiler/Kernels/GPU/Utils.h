#ifndef MULTI_DEVICE_MODEL_COMPILER_KERNELS_GPU_UTILS_H_
#define MULTI_DEVICE_MODEL_COMPILER_KERNELS_GPU_UTILS_H_

#include "multi-device-model-compiler/Kernels/GPU/Ops.h"

template <typename T_OUT, typename T_IN>
MLIR_GPU_OPS_EXPORT void invokeCudaCast(T_OUT *dst, T_IN const *const src,
                                        const size_t size, cudaStream_t stream);

#endif