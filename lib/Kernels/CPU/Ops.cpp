#include "multi-device-model-compiler/Kernels/CPU/Ops.h"

#include "dnnl.hpp"

#include "xxhash.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <unordered_map>

using namespace dnnl;

static engine cpuEngine;
static stream cpuStream;
static bool fastFlag;

void cpuOpsInit(bool fastMathFlag) {
  cpuEngine = engine(engine::kind::cpu, 0);
  cpuStream = stream(cpuEngine);
  fastFlag = fastMathFlag;
}

static std::unordered_map<int64_t, matmul> matmulMap;

extern "C" MLIR_CPU_OPS_EXPORT void mcpuMatmul(float *input, float *weight,
                                               float *bias, float *output,
                                               int64_t M, int64_t N,
                                               int64_t K) {
  std::array<int64_t, 3> keys{M, N, K};
  int64_t matmulKey = XXH3_64bits(keys.data(), keys.size() * sizeof(int64_t));
  matmul matmulPrim;

  memory::dims inputDims = {M, K}, weightDims = {K, N}, biasDims = {1, N},
               outputDims = {M, N};
  auto inputMd = memory::desc(inputDims, memory::data_type::f32,
                              memory::format_tag::ab),
       weightMd = memory::desc(weightDims, memory::data_type::f32,
                               memory::format_tag::ab),
       biasMd = memory::desc(biasDims, memory::data_type::f32,
                             memory::format_tag::ab),
       outputMd = memory::desc(outputDims, memory::data_type::f32,
                               memory::format_tag::ab);

  if (matmulMap.count(matmulKey)) {
    matmulPrim = matmulMap[matmulKey];
  } else {
    primitive_attr matmulAttr;
    if (fastFlag) {
      matmulAttr.set_fpmath_mode(fpmath_mode::any);
    }
    auto matmulPd = matmul::primitive_desc(cpuEngine, inputMd, weightMd, biasMd,
                                           outputMd, matmulAttr);
    matmulPrim = matmul(matmulPd);

    matmulMap.emplace(matmulKey, matmulPrim);
  }

  auto inputMem = memory(inputMd, cpuEngine, input),
       weightMem = memory(weightMd, cpuEngine, weight),
       biasMem = memory(biasMd, cpuEngine, bias),
       outputMem = memory(outputMd, cpuEngine, output);

  std::unordered_map<int, memory> matmulArgs;
  matmulArgs.emplace(DNNL_ARG_SRC, inputMem);
  matmulArgs.emplace(DNNL_ARG_WEIGHTS, weightMem);
  matmulArgs.emplace(DNNL_ARG_BIAS, biasMem);
  matmulArgs.emplace(DNNL_ARG_DST, outputMem);

  matmulPrim.execute(cpuStream, matmulArgs);
  cpuStream.wait();
}

static std::unordered_map<int64_t, convolution_forward> conv2dMap;

extern "C" MLIR_CPU_OPS_EXPORT void
mcpuConv2d(float *input, float *weight, float *bias, float *output,
           float *postAdd, int64_t N, int64_t IC, int64_t H, int64_t W,
           int64_t OC, int64_t KH, int64_t KW, int64_t OH, int64_t OW,
           int64_t PHL, int64_t PWL, int64_t PHR, int64_t PWR, int64_t SH,
           int64_t SW, int64_t DH, int64_t DW, bool hasPostAdd,
           bool hasContainRelu) {
  std::array<int64_t, 10> keys{N,   IC, H,  OC,         KH,
                               PHL, SH, DH, hasPostAdd, hasContainRelu};
  int64_t convKey = XXH3_64bits(keys.data(), keys.size() * sizeof(int64_t));

  convolution_forward convPrim;

  memory::dims inputDims = {N, IC, H, W}, weightDims = {OC, IC, KH, KW},
               biasDims = {OC}, outputDims = {N, OC, OH, OW};

  memory::dims stridesDims = {SH, SW}, paddindDimsL = {PHL, PWL},
               paddingDimsR = {PHR, PWR}, dialationDims = {DH - 1, DW - 1};

  auto inputMd = memory::desc(inputDims, memory::data_type::f32,
                              memory::format_tag::nchw),
       weightMd = memory::desc(weightDims, memory::data_type::f32,
                               memory::format_tag::oihw),
       biasMd = memory::desc(biasDims, memory::data_type::f32,
                             memory::format_tag::a),
       outputMd = memory::desc(outputDims, memory::data_type::f32,
                               memory::format_tag::nchw);

  if (conv2dMap.count(convKey)) {
    convPrim = conv2dMap[convKey];
  } else {
    primitive_attr convAttr;
    if (fastFlag) {
      convAttr.set_fpmath_mode(fpmath_mode::any);
    }
    if (hasContainRelu) {
      post_ops postOp;
      postOp.append_eltwise(algorithm::eltwise_relu, 0, 0);
      convAttr.set_post_ops(postOp);
    }
    if (hasPostAdd) {
      post_ops postOp;
      postOp.append_binary(algorithm::binary_add, outputMd);
      convAttr.set_post_ops(postOp);
    }
    auto convPd = convolution_forward::primitive_desc(
        cpuEngine, prop_kind::forward_inference, algorithm::convolution_auto,
        inputMd, weightMd, biasMd, outputMd, stridesDims, dialationDims,
        paddindDimsL, paddingDimsR, convAttr);

    convPrim = convolution_forward(convPd);
    conv2dMap.emplace(convKey, convPrim);
  }

  auto inputMem = memory(inputMd, cpuEngine, input),
       weightMem = memory(weightMd, cpuEngine, weight),
       biasMem = memory(biasMd, cpuEngine, bias),
       outputMem = memory(outputMd, cpuEngine, output);

  std::unordered_map<int, memory> convArgs;
  convArgs.emplace(DNNL_ARG_SRC, inputMem);
  convArgs.emplace(DNNL_ARG_WEIGHTS, weightMem);
  convArgs.emplace(DNNL_ARG_BIAS, biasMem);
  convArgs.emplace(DNNL_ARG_DST, outputMem);
  if (hasPostAdd) {
    auto postAddMem = memory(outputMd, cpuEngine, postAdd);
    convArgs.emplace(DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                     postAddMem);
  }

  convPrim.execute(cpuStream, convArgs);
  cpuStream.wait();
}

static std::unordered_map<int64_t, pooling_forward> pool2dMap;

extern "C" MLIR_CPU_OPS_EXPORT void
mcpuPool2d(float *input, float *output, int64_t N, int64_t C, int64_t H,
           int64_t W, int64_t OH, int64_t OW, int64_t KH, int64_t KW,
           int64_t PHL, int64_t PWL, int64_t PHR, int64_t PWR, int64_t SH,
           int64_t SW, int64_t method) {
  std::array<int64_t, 8> keys{N, C, H, OH, KH, PHL, SH, method};
  int64_t poolKey = XXH3_64bits(keys.data(), keys.size() * sizeof(int64_t));

  pooling_forward poolPrim;

  memory::dims inputDims = {N, C, H, W}, outputDims = {N, C, OH, OW};

  memory::dims kernelDims = {KH, KW}, strideDims = {SH, SW},
               paddingDimsL = {PHL, PWL}, paddingDimsR = {PHR, PWR},
               dilationDims = {0, 0};

  auto inputMd = memory::desc(inputDims, memory::data_type::f32,
                              memory::format_tag::nchw),
       outputMd = memory::desc(outputDims, memory::data_type::f32,
                               memory::format_tag::nchw);

  if (pool2dMap.count(poolKey)) {
    poolPrim = pool2dMap[poolKey];
  } else {
    auto algorithm = algorithm::pooling_max;
    if (method == 1) {
      algorithm = algorithm::pooling_avg_include_padding;
    }
    auto poolPd = pooling_forward::primitive_desc(
        cpuEngine, prop_kind::forward_inference, algorithm, inputMd, outputMd,
        strideDims, kernelDims, dilationDims, paddingDimsL, paddingDimsR);

    poolPrim = pooling_forward(poolPd);

    pool2dMap.emplace(poolKey, poolPrim);
  }

  auto inputMem = memory(inputMd, cpuEngine, input),
       outputMem = memory(outputMd, cpuEngine, output);

  std::unordered_map<int, memory> poolArgs;
  poolArgs.emplace(DNNL_ARG_SRC, inputMem);
  poolArgs.emplace(DNNL_ARG_DST, outputMem);

  poolPrim.execute(cpuStream, poolArgs);
  cpuStream.wait();
}