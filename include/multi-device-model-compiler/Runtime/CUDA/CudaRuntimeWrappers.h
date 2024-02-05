#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <stdio.h>

#include "cuda.h"
#include "cuda_bf16.h"
#include "cuda_fp16.h"

#ifdef MLIR_ENABLE_CUDA_CUSPARSE
#include "cusparse.h"
#ifdef MLIR_ENABLE_CUDA_CUSPARSELT
#include "cusparseLt.h"
#endif // MLIR_ENABLE_CUDA_CUSPARSELT
#endif // MLIR_ENABLE_CUDA_CUSPARSE

#ifdef _WIN32
#define MLIR_CUDA_WRAPPERS_EXPORT __declspec(dllexport)
#else
#define MLIR_CUDA_WRAPPERS_EXPORT
#endif // _WIN32

extern "C" {
MLIR_CUDA_WRAPPERS_EXPORT void mgpuModuleFileLoad(const char *fname);
MLIR_CUDA_WRAPPERS_EXPORT void mgpuModuleUnload();
MLIR_CUDA_WRAPPERS_EXPORT CUfunction mgpuModuleGetFunction(const char *name);
MLIR_CUDA_WRAPPERS_EXPORT void
mgpuLaunchKernel(CUfunction function, intptr_t gridX, intptr_t gridY,
                 intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                 intptr_t blockZ, int32_t smem, CUstream stream, void **params,
                 void **extra);
MLIR_CUDA_WRAPPERS_EXPORT CUstream mgpuStreamCreate();
MLIR_CUDA_WRAPPERS_EXPORT void mgpuStreamDestroy(CUstream stream);
MLIR_CUDA_WRAPPERS_EXPORT void mgpuStreamWaitEvent(CUstream stream,
                                                   CUevent event);
MLIR_CUDA_WRAPPERS_EXPORT CUevent mgpuEventCreate();
MLIR_CUDA_WRAPPERS_EXPORT CUevent mgpuEventEnableTimeCreate();
MLIR_CUDA_WRAPPERS_EXPORT CUevent mgpuEventCreateWithStream(CUstream stream);
MLIR_CUDA_WRAPPERS_EXPORT void mgpuEventDestroy(CUevent event);
MLIR_CUDA_WRAPPERS_EXPORT void mgpuEventSynchronize(CUevent event);
MLIR_CUDA_WRAPPERS_EXPORT void mgpuEventRecord(CUevent event, CUstream stream);
MLIR_CUDA_WRAPPERS_EXPORT float mgpuEventElapsedTime(CUevent begin,
                                                     CUevent end);
void *mgpuMemAlloc(uint64_t sizeBytes, CUstream /*stream*/);
void mgpuMemFree(void *ptr, CUstream /*stream*/);
void mgpuMemcpy(void *dst, void *src, size_t sizeBytes, CUstream stream);
void mgpuMemset32(void *dst, unsigned int value, size_t count, CUstream stream);
void mgpuMemset16(void *dst, unsigned short value, size_t count,
                  CUstream stream);
}