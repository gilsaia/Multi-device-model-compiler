#include "multi-device-model-compiler/Utils/CompileOptions.h"
#include "ExternalUtil.h"

#include <string>

namespace multi_device {
llvm::cl::OptionCategory MultiDeviceCompileOptions("Common option for compile",
                                                   "");
llvm::cl::OptionCategory MultiDeviceLibGenOptions("Generate Library Options",
                                                  "");

llvm::cl::opt<bool> preserveOptIR("preserveOptIR",
                                  llvm::cl::desc("don't delete opt ir files."),
                                  llvm::cl::init(false),
                                  llvm::cl::cat(MultiDeviceLibGenOptions));

llvm::cl::opt<bool> preserveObject("preserveObject",
                                   llvm::cl::desc("don't delete object files."),
                                   llvm::cl::init(false),
                                   llvm::cl::cat(MultiDeviceLibGenOptions));

llvm::cl::opt<std::string>
    mtriple("mtriple", llvm::cl::desc("Override target triple for module"),
            llvm::cl::value_desc("LLVM target triple"),
            llvm::cl::cat(MultiDeviceCompileOptions), llvm::cl::ValueRequired);

llvm::cl::opt<std::string>
    mcpu("mcpu", llvm::cl::desc("Target cpu"),
         llvm::cl::value_desc("Target a specific CPU type"),
         llvm::cl::cat(MultiDeviceCompileOptions), llvm::cl::ValueRequired);

llvm::cl::opt<std::string>
    march("march", llvm::cl::desc("Target architecture to generate code for"),
          llvm::cl::value_desc("Target a specific architecture type"),
          llvm::cl::cat(MultiDeviceCompileOptions), llvm::cl::ValueRequired);

llvm::cl::opt<OptLevel> OptimizationLevel(
    llvm::cl::desc("Levels:"),
    llvm::cl::values(clEnumVal(O0, "Optimization level 0,"),
                     clEnumVal(O1, "Optimization level 1,"),
                     clEnumVal(O2, "Optimization level 2 (default),"),
                     clEnumVal(O3, "Optimization level 3.")),
    llvm::cl::init(O2), llvm::cl::cat(MultiDeviceCompileOptions));

std::string getTargetArch() { return (march != "") ? "--march=" + march : ""; }

std::string getTargetTriple() {
  std::string targetOptions = "";
  // Command cannot tolerate extra spaces. Add only when needed.
  if (mtriple != "")
    targetOptions = "--mtriple=" + mtriple;
  else if (kDefaultTriple != "")
    targetOptions = "--mtriple=" + kDefaultTriple;
  return targetOptions;
}

std::string getTargetCPU() { return (mcpu != "") ? "--mcpu=" + mcpu : ""; }

std::string getOptimizationLevel() {
  switch (OptimizationLevel) {
  case OptLevel::O0:
    return "-O0";
  case OptLevel::O1:
    return "-O1";
  case OptLevel::O2:
    return "-O2";
  case OptLevel::O3:
    return "-O3";
  }
  llvm_unreachable("Unexpected optimization level");
  return "";
}

} // namespace multi_device