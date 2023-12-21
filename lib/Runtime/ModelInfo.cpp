#include "multi-device-model-compiler/Runtime/ModelInfo.h"

#include "llvm/Support/CommandLine.h"

#include <string>

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
} // namespace multi_device
