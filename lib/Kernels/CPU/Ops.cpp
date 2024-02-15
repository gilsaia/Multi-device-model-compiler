#include "multi-device-model-compiler/Kernels/CPU/Ops.h"

#include "dnnl.hpp"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"

#include <unordered_map>

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

using namespace dnnl;

static engine cpuEngine;
static stream cpuStream;

void cpuOpsInit() {
  cpuEngine = engine(engine::kind::cpu, 0);
  cpuStream = stream(cpuEngine);
}

static llvm::DenseMap<llvm::SmallVector<int64_t>, matmul::primitive_desc>
    matmulMap;

extern "C" MLIR_CPU_OPS_EXPORT void mcpuMatmul(float *input, float *weight,
                                               float *bias, float *output,
                                               int64_t M, int64_t N,
                                               int64_t K) {
  llvm::SmallVector<int64_t> matmulKey{M, N, K};
  matmul::primitive_desc matmulPd;

  memory::dims inputDims = {M, K}, weightDims = {K, N}, biasDims = {N},
               outputDims = {M, N};
  auto inputMd = memory::desc(inputDims, memory::data_type::f32,
                              memory::format_tag::ab),
       weightMd = memory::desc(weightDims, memory::data_type::f32,
                               memory::format_tag::ab),
       biasMd = memory::desc(biasDims, memory::data_type::f32,
                             memory::format_tag::a),
       outputMd = memory::desc(outputDims, memory::data_type::f32,
                               memory::format_tag::ab);

  if (matmulMap.count(matmulKey)) {
    matmulPd = matmulMap[matmulKey];
  } else {
    primitive_attr matmulAttr;
    matmulPd = matmul::primitive_desc(cpuEngine, inputMd, weightMd, biasMd,
                                      outputMd, matmulAttr);
    matmulMap.insert({matmulKey, matmulPd});
  }
  auto matmulPrim = matmul(matmulPd);

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