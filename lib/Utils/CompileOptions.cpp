#include "multi-device-model-compiler/Utils/CompileOptions.h"
#include "ExternalUtil.h"

#include <string>

namespace multi_device {
llvm::cl::OptionCategory MultiDeviceCompileOptions("Common option for compile",
                                                   "");
llvm::cl::OptionCategory MultiDeviceLibGenOptions("Generate Library Options",
                                                  "");

llvm::cl::opt<bool> preserveOptIR("saveOptIR",
                                  llvm::cl::desc("don't delete opt ir files."),
                                  llvm::cl::init(false),
                                  llvm::cl::cat(MultiDeviceLibGenOptions));

llvm::cl::opt<bool> preserveObject("saveObject",
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

llvm::cl::opt<TargetDevice>
    Device("device", llvm::cl::desc("Set target device:"),
           llvm::cl::values(clEnumVal(CPU, "Target device cpu (default),"),
                            clEnumVal(GPU, "Target device gpu (cuda),"),
                            clEnumVal(TPU, "Target device tpu.")),
           llvm::cl::init(CPU), llvm::cl::cat(MultiDeviceCompileOptions));

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

std::string getLibraryExt() {
  switch (Device) {
  case TargetDevice::CPU:
  case TargetDevice::GPU:
    return ".so";
  case TargetDevice::TPU:
    return ".bmodel";
  }
  llvm_unreachable("Unexpected target device");
  return "";
}

#ifdef MULTI_DEVICE_CUDA_ENABLE

llvm::cl::opt<bool> preservePTX("savePTX",
                                llvm::cl::desc("don't delete ptx files."),
                                llvm::cl::init(false),
                                llvm::cl::cat(MultiDeviceLibGenOptions));

llvm::cl::opt<std::string> ktriple("ktriple",
                                   llvm::cl::desc("Target triple for gpu"),
                                   llvm::cl::value_desc("GPU target triple"),
                                   llvm::cl::init("nvptx64-nvidia-cuda"),
                                   llvm::cl::cat(MultiDeviceCompileOptions),
                                   llvm::cl::ValueRequired);

llvm::cl::opt<std::string>
    kcpu("kcpu", llvm::cl::desc("Target gpu"),
         llvm::cl::value_desc("Target specific GPU type."),
         llvm::cl::init("sm_70"), llvm::cl::cat(MultiDeviceCompileOptions),
         llvm::cl::ValueRequired);

llvm::cl::opt<std::string>
    karch("karch", llvm::cl::desc("Target architecture to generate code for"),
          llvm::cl::value_desc("Target a specific architecture gpu type"),
          llvm::cl::init("nvptx64"), llvm::cl::cat(MultiDeviceCompileOptions),
          llvm::cl::ValueRequired);

llvm::cl::opt<std::string> kattr("kattr", llvm::cl::desc("Target ptx version"),
                                 llvm::cl::value_desc("Target PTX version"),
                                 llvm::cl::init("+ptx76"),
                                 llvm::cl::cat(MultiDeviceCompileOptions),
                                 llvm::cl::ValueRequired);

std::string getKernelTriple() { return "--mtriple=" + ktriple; }

std::string getKernelArch() { return "--march=" + karch; }

std::string getPTXArch() { return "-arch=" + kcpu; }

std::string getKernelCPU() { return "--mcpu=" + kcpu; }

std::string getKernelAttr() { return "--mattr=" + kattr; }

#endif

} // namespace multi_device