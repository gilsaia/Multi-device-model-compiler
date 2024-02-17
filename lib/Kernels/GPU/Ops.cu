#include "multi-device-model-compiler/Kernels/GPU/Ops.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/raw_ostream.h"

#include "cublasLt.h"

#include <stdexcept>

static cublasLtHandle_t ltHandle;

static void *workspace;
static size_t workspaceSize = 1024 * 1024 * 16;

inline void checkCudaStatus(cudaError_t status) {
  if (status != cudaSuccess) {
    printf("cuda API failed with status %d: %s\n", status,
           cudaGetErrorString(status));
    throw std::logic_error("cuda API failed");
  }
}

inline void checkCublasStatus(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("cuBLAS API failed with status %d\n", status);
    throw std::logic_error("cuBLAS API failed");
  }
}

namespace llvm {
struct SmallVectorHasher {
  std::size_t operator()(const SmallVector<int64_t> &vec) const {
    return hash_combine_range(vec.begin(), vec.end());
  }
};
template <> struct DenseMapInfo<SmallVector<int64_t>> {
  static inline SmallVector<int64_t> getEmptyKey() {
    SmallVector<int64_t> EmptyKey{-1};
    return EmptyKey;
  }
  static inline SmallVector<int64_t> getTombstoneKey() {
    SmallVector<int64_t> TombstoneKey{-2};
    return TombstoneKey;
  }
  static unsigned getHashValue(const SmallVector<int64_t> &vec) {
    return SmallVectorHasher()(vec);
  }
  static bool isEqual(const SmallVector<int64_t> &lhs,
                      const SmallVector<int64_t> &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

void gpuOpsInit() {
  checkCublasStatus(cublasLtCreate(&ltHandle));
  checkCudaStatus(cudaMalloc(&workspace, workspaceSize));
}

void gpuOpsDeinit() {
  checkCublasStatus(cublasLtDestroy(ltHandle));
  checkCudaStatus(cudaFree(workspace));
}

static llvm::DenseMap<llvm::SmallVector<int64_t>,
                      cublasLtMatmulHeuristicResult_t>
    matmulMap;

extern "C" MLIR_GPU_OPS_EXPORT void mgpuMatmul(float *input, float *weight,
                                               float *bias, float *output,
                                               int64_t M, int64_t N, int64_t K,
                                               cudaStream_t stream) {
  llvm::SmallVector<int64_t> matmulKey{M, N, K};
  cublasLtMatmulDesc_t operationDesc = NULL;
  cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Ddesc = NULL;

  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  cublasLtMatmulPreference_t preference = NULL;
  int returnedResults = 0;

  auto epilogue = CUBLASLT_EPILOGUE_BIAS;
  checkCublasStatus(cublasLtMatmulDescCreate(
      &operationDesc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
      sizeof(epilogue)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias,
      sizeof(float *)));

  checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, N, K, N));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, K, M, K));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_32F, N, M, N));
  if (matmulMap.count(matmulKey)) {
    heuristicResult = matmulMap[matmulKey];
  } else {
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize,
        sizeof(workspaceSize)));

    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, operationDesc, Adesc, Bdesc, Ddesc, Ddesc, preference, 1,
        &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
      checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }
    matmulMap.insert({matmulKey, heuristicResult});
  }

  float alpha = 1, beta = 0;
  checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, &alpha, weight,
                                   Adesc, input, Bdesc, &beta, output, Ddesc,
                                   output, Ddesc, &heuristicResult.algo,
                                   workspace, workspaceSize, stream));

  if (preference)
    checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
  if (Ddesc)
    checkCublasStatus(cublasLtMatrixLayoutDestroy(Ddesc));
  if (Bdesc)
    checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
  if (Adesc)
    checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
  if (operationDesc)
    checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}