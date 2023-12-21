#ifndef MULTI_DEVICE_MODEL_COMPILER_RUNTIME_MODELINFO_H_
#define MULTI_DEVICE_MODEL_COMPILER_RUNTIME_MODELINFO_H_

#include "llvm/Support/CommandLine.h"

#include <vector>

namespace multi_device {

extern llvm::cl::OptionCategory MultiDeviceModelOptions;

extern llvm::cl::opt<std::string> InputShapes;
extern llvm::cl::opt<std::string> OutputShapes;

struct ModelInfo {
  static ModelInfo GetModelInfo();
  int InputNums();
  int OutputNums();
  std::vector<int> &GetInputSize(int idx);
  std::vector<int> &GetOutputSize(int idx);

private:
  std::vector<std::vector<int>> sizes;
};

} // namespace multi_device

#endif