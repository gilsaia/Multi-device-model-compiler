#include "multi-device-model-compiler/Kernels/GPU/Ops.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/raw_ostream.h"

#include "cublasLt.h"
#include "cudnn.h"

#include "xxhash.h"

#include <float.h>
#include <stdexcept>

static cublasLtHandle_t ltHandle;
static cudnnHandle_t cudnnHandle;

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

inline void checkCudnnStatus(cudnnStatus_t status) {
  if (status != CUDNN_STATUS_SUCCESS) {
    printf("cuDNN API failed with status %d\n", status);
    throw std::logic_error("cuDNN API failed");
  }
}

void gpuOpsInit() {
  checkCublasStatus(cublasLtCreate(&ltHandle));
  checkCudaStatus(cudaMalloc(&workspace, workspaceSize));
  checkCudnnStatus(cudnnCreate(&cudnnHandle));
}

void gpuOpsDeinit() {
  checkCudnnStatus(cudnnDestroy(cudnnHandle));
  checkCublasStatus(cublasLtDestroy(ltHandle));
  checkCudaStatus(cudaFree(workspace));
}

static std::unordered_map<int64_t, cublasLtMatmulHeuristicResult_t> matmulMap;

extern "C" MLIR_GPU_OPS_EXPORT void mgpuMatmul(float *input, float *weight,
                                               float *bias, float *output,
                                               int64_t M, int64_t N, int64_t K,
                                               cudaStream_t stream) {
  // llvm::SmallVector<int64_t> matmulKey{M, N, K};
  std::array<int64_t, 3> keys{M, N, K};
  int64_t matmulKey = XXH3_64bits(keys.data(), keys.size() * sizeof(int64_t));
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
    matmulMap.emplace(matmulKey, heuristicResult);
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

static std::unordered_map<int64_t, cudnnConvolutionFwdAlgo_t> conv2dMap;

extern "C" MLIR_GPU_OPS_EXPORT void
mgpuConv2d(float *input, float *weight, float *bias, float *output,
           float *postAdd, int64_t N, int64_t IC, int64_t H, int64_t W,
           int64_t OC, int64_t KH, int64_t KW, int64_t OH, int64_t OW,
           int64_t PHL, int64_t PWL, int64_t PHR, int64_t PWR, int64_t SH,
           int64_t SW, int64_t DH, int64_t DW, bool hasPostAdd,
           bool hasContainRelu, cudaStream_t stream) {
  std::array<int64_t, 10> keys{N,   IC, H,  OC,         KH,
                               PHL, SH, DH, hasPostAdd, hasContainRelu};
  int64_t convKey = XXH3_64bits(keys.data(), keys.size() * sizeof(int64_t));

  cudnnSetStream(cudnnHandle, stream);

  cudnnConvolutionDescriptor_t convDesc;
  checkCudnnStatus(cudnnCreateConvolutionDescriptor(&convDesc));
  checkCudnnStatus(cudnnSetConvolution2dDescriptor(
      convDesc, PHL, PWL, SH, SW, DH, DW, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
  checkCudnnStatus(cudnnSetConvolutionMathType(
      convDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));

  cudnnTensorDescriptor_t inputDesc, biasDesc, outputDesc, addDesc;
  float *add;
  checkCudnnStatus(cudnnCreateTensorDescriptor(&inputDesc));
  checkCudnnStatus(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT, N, IC, H, W));
  checkCudnnStatus(cudnnCreateTensorDescriptor(&biasDesc));
  checkCudnnStatus(cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT, 1, OC, 1, 1));
  checkCudnnStatus(cudnnCreateTensorDescriptor(&outputDesc));
  checkCudnnStatus(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT, N, OC, OH, OW));
  if (hasPostAdd) {
    checkCudnnStatus(cudnnCreateTensorDescriptor(&addDesc));
    checkCudnnStatus(cudnnSetTensor4dDescriptor(
        addDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, OC, OH, OW));
    add = postAdd;
  } else {
    addDesc = outputDesc;
    add = output;
  }

  cudnnFilterDescriptor_t weightDesc;
  checkCudnnStatus(cudnnCreateFilterDescriptor(&weightDesc));
  checkCudnnStatus(cudnnSetFilter4dDescriptor(
      weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, OC, IC, KH, KW));

  cudnnActivationDescriptor_t actDesc;
  checkCudnnStatus(cudnnCreateActivationDescriptor(&actDesc));
  checkCudnnStatus(cudnnSetActivationDescriptor(
      actDesc,
      hasContainRelu ? CUDNN_ACTIVATION_RELU : CUDNN_ACTIVATION_IDENTITY,
      CUDNN_NOT_PROPAGATE_NAN, DBL_MAX));

  cudnnConvolutionFwdAlgo_t algo;
  if (conv2dMap.count(convKey)) {
    algo = conv2dMap[convKey];
  } else {
    algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    if (hasContainRelu) {
      int algoMax;
      checkCudnnStatus(
          cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle, &algoMax));
      std::vector<cudnnConvolutionFwdAlgoPerf_t> perfs(algoMax);
      int algoFind;
      checkCudnnStatus(cudnnFindConvolutionForwardAlgorithm(
          cudnnHandle, inputDesc, weightDesc, convDesc, outputDesc, algoMax,
          &algoFind, perfs.data()));
      perfs.resize(algoFind);
      if (algoFind < 1) {
        checkCudnnStatus(CUDNN_STATUS_NOT_SUPPORTED);
      }
      algo = perfs[0].algo;
    }

    size_t nWorkSpaceSize;
    checkCudnnStatus(cudnnGetConvolutionForwardWorkspaceSize(
        cudnnHandle, inputDesc, weightDesc, convDesc, outputDesc, algo,
        &nWorkSpaceSize));
    if (nWorkSpaceSize > workspaceSize) {
      cudaFree(workspace);
      cudaMalloc(&workspace, nWorkSpaceSize);
      workspaceSize = nWorkSpaceSize;
    }
    conv2dMap.emplace(convKey, algo);
  }

  float alpha1 = 1, alpha2 = hasPostAdd ? 1 : 0;

  checkCudnnStatus(cudnnConvolutionBiasActivationForward(
      cudnnHandle, &alpha1, inputDesc, input, weightDesc, weight, convDesc,
      algo, workspace, workspaceSize, &alpha2, addDesc, add, biasDesc, bias,
      actDesc, outputDesc, output));

  cudnnDestroyConvolutionDescriptor(convDesc);
  cudnnDestroyTensorDescriptor(inputDesc);
  cudnnDestroyTensorDescriptor(biasDesc);
  cudnnDestroyTensorDescriptor(outputDesc);
  cudnnDestroyFilterDescriptor(weightDesc);
  cudnnDestroyActivationDescriptor(actDesc);
  if (hasPostAdd) {
    cudnnDestroyTensorDescriptor(addDesc);
  }
}

extern "C" MLIR_GPU_OPS_EXPORT void
mgpuPool2d(float *input, float *output, int64_t N, int64_t C, int64_t H,
           int64_t W, int64_t OH, int64_t OW, int64_t KH, int64_t KW,
           int64_t PHL, int64_t PWL, int64_t PHR, int64_t PWR, int64_t SH,
           int64_t SW, int64_t method /* 0 - max, 1 - avg */,
           cudaStream_t stream) {
  cudnnSetStream(cudnnHandle, stream);

  cudnnPoolingDescriptor_t poolDesc;
  cudnnPoolingMode_t mode = (method == 0)
                                ? CUDNN_POOLING_MAX
                                : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
  checkCudnnStatus(cudnnCreatePoolingDescriptor(&poolDesc));
  checkCudnnStatus(cudnnSetPooling2dDescriptor(
      poolDesc, mode, CUDNN_NOT_PROPAGATE_NAN, KH, KW, PHL, PWL, SH, SW));

  cudnnTensorDescriptor_t inputDesc, outputDesc;
  checkCudnnStatus(cudnnCreateTensorDescriptor(&inputDesc));
  checkCudnnStatus(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT, N, C, H, W));
  checkCudnnStatus(cudnnCreateTensorDescriptor(&outputDesc));
  checkCudnnStatus(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT, N, C, OH, OW));

  float alpha = 1, beta = 0;

  checkCudnnStatus(cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, inputDesc,
                                       input, &beta, outputDesc, output));

  cudnnDestroyTensorDescriptor(inputDesc);
  cudnnDestroyTensorDescriptor(outputDesc);
  cudnnDestroyPoolingDescriptor(poolDesc);
}