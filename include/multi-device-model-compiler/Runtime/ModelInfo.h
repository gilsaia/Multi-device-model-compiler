#ifndef MULTI_DEVICE_MODEL_COMPILER_RUNTIME_MODELINFO_H_
#define MULTI_DEVICE_MODEL_COMPILER_RUNTIME_MODELINFO_H_

#include "llvm/Support/CommandLine.h"

#include <vector>

namespace multi_device {

extern llvm::cl::OptionCategory MultiDeviceModelOptions;

extern llvm::cl::opt<std::string> InputShapes;
extern llvm::cl::opt<std::string> OutputShapes;

struct ModelInfo {
  static ModelInfo *ParseModelInfo();
  size_t InputNums();
  size_t OutputNums();
  size_t GetDim();
  std::vector<size_t> &GetInputSize(int idx);
  std::vector<size_t> &GetOutputSize(int idx);

private:
  std::vector<std::vector<size_t>> inputSizes, outputSizes;
  size_t dim;
};

} // namespace multi_device

#endif