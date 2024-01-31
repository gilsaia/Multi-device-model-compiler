#ifndef MULTI_DEVICE_MODEL_COMPILER_RUNTIME_MODELINFO_H_
#define MULTI_DEVICE_MODEL_COMPILER_RUNTIME_MODELINFO_H_

#include "llvm/Support/CommandLine.h"

#include <vector>

namespace multi_device {

extern llvm::cl::OptionCategory MultiDeviceModelOptions;

extern llvm::cl::opt<std::string> InputShapes;
extern llvm::cl::opt<std::string> OutputShapes;
extern llvm::cl::list<std::string> InputData;

struct ModelInfo {
  static ModelInfo *ParseModelInfo();
  size_t InputNums();
  size_t OutputNums();
  size_t GetInputNumElements(int idx);
  size_t GetOutputNumElements(int idx);
  std::vector<size_t> &GetInputSize(int idx);
  std::vector<size_t> &GetOutputSize(int idx);
  std::string GetFile(int idx);
  bool ExistFile();

private:
  std::vector<std::vector<size_t>> inputSizes, outputSizes;
  std::vector<std::string> inputData;
};

} // namespace multi_device

#endif