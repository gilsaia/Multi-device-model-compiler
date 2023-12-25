#include "multi-device-model-compiler/Runtime/ModelInfo.h"

#include "llvm/Support/CommandLine.h"

#include <string>
#include <vector>

namespace multi_device {

llvm::cl::OptionCategory MultiDeviceModelOptions("Options use for model info",
                                                 "");

llvm::cl::opt<std::string>
    InputShapes("input",
                llvm::cl::desc("Input shape,Use like 3_640_640,3_640_640"),
                llvm::cl::init("3_640_640,3_640_640"), llvm::cl::ValueRequired,
                llvm::cl::cat(MultiDeviceModelOptions));

llvm::cl::opt<std::string>
    OutputShapes("output", llvm::cl::desc("Output shape,Use like 3_640_640"),
                 llvm::cl::init("3_640_640"), llvm::cl::ValueRequired,
                 llvm::cl::cat(MultiDeviceModelOptions));

std::vector<size_t> parseOneShape(std::string &&input) {
  std::vector<size_t> shape;
  size_t pos = input.find('_'), start = 0;
  while (pos != std::string::npos) {
    shape.push_back(std::stoul(input.substr(start, pos - start)));
    start = pos + 1;
    pos = input.find('_', pos + 1);
  }
  shape.push_back(std::stoul(input.substr(start)));
  return shape;
}

std::vector<std::vector<size_t>> parseShapes(std::string &str) {
  std::vector<std::vector<size_t>> shapes;
  size_t pos = str.find(','), start = 0;
  while (pos != std::string::npos) {
    shapes.emplace_back(parseOneShape(str.substr(start, pos - start)));
    start = pos + 1;
    pos = str.find(',', pos + 1);
  }
  shapes.emplace_back(parseOneShape(str.substr(start)));
  return shapes;
}

ModelInfo *ModelInfo::ParseModelInfo() {
  ModelInfo *info = new ModelInfo();
  info->inputSizes = parseShapes(InputShapes);
  info->outputSizes = parseShapes(OutputShapes);
  size_t dim = 0;
  for (auto &input : info->inputSizes) {
    dim = std::max(dim, input.size());
  }
  for (auto &output : info->outputSizes) {
    dim = std::max(dim, output.size());
  }
  info->dim = dim;
  for (auto &input : info->inputSizes) {
    while (input.size() < dim) {
      input.emplace(input.begin(), 1);
    }
  }
  for (auto &output : info->outputSizes) {
    while (output.size() < dim) {
      output.emplace(output.begin(), 1);
    }
  }
  return info;
}

size_t ModelInfo::InputNums() { return inputSizes.size(); }
size_t ModelInfo::OutputNums() { return outputSizes.size(); }

std::vector<size_t> &ModelInfo::GetInputSize(int idx) {
  return inputSizes[idx];
}
std::vector<size_t> &ModelInfo::GetOutputSize(int idx) {
  return outputSizes[idx];
}

size_t ModelInfo::GetDim() { return dim; }
} // namespace multi_device
