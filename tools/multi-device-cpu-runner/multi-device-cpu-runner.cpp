#include "multi-device-model-compiler/Runtime/ModelInfo.h"
#include "multi-device-model-compiler/Runtime/RuntimeUtil.h"
#include "multi-device-model-compiler/Runtime/TensorDescripter.h"

#ifndef MULTI_DEVICE_PLAIN_RUNTIME
#include "multi-device-model-compiler/Kernels/CPU/Ops.h"
#endif

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

#include <chrono>
#include <dlfcn.h>
#include <string>

llvm::cl::OptionCategory
    MultiDeviceCpuRunnerOptions("Options use for cpu runner", "");

llvm::cl::opt<std::string> LibPath(llvm::cl::value_desc("Model Library Path"),
                                   llvm::cl::desc("<lib>"), llvm::cl::Required,
                                   llvm::cl::Positional,
                                   llvm::cl::cat(MultiDeviceCpuRunnerOptions));
llvm::cl::opt<std::string>
    RunFuncName("func", llvm::cl::desc("Model Run Func Name"),
                llvm::cl::ValueRequired,
                llvm::cl::init("_mlir_ciface_main_graph"),
                llvm::cl::cat(MultiDeviceCpuRunnerOptions));

llvm::cl::opt<std::string>
    OutputName("out-data", llvm::cl::desc("Output file name"),
               llvm::cl::ValueRequired, llvm::cl::init(""),
               llvm::cl::cat(MultiDeviceCpuRunnerOptions));

llvm::cl::opt<int> RerunTimes("rerun", llvm::cl::desc("Rerun time"),
                              llvm::cl::init(0), llvm::cl::ValueRequired,
                              llvm::cl::cat(MultiDeviceCpuRunnerOptions));

llvm::cl::opt<bool> FastMode("fast",
                             llvm::cl::desc("Fast mode,with precision loss"),
                             llvm::cl::cat(MultiDeviceCpuRunnerOptions));

using namespace multi_device;

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::HideUnrelatedOptions(
      {&MultiDeviceCpuRunnerOptions, &MultiDeviceModelOptions});

  if (!llvm::cl::ParseCommandLineOptions(argc, argv,
                                         "Multi device cpu runner\n")) {
    llvm::errs() << "Failed to parse options\n";
    return 1;
  }

#ifndef MULTI_DEVICE_PLAIN_RUNTIME
  cpuOpsInit(FastMode);
#endif

  ModelInfo *info = ModelInfo::ParseModelInfo();
  auto params = GetParamsVec(info);

  void *handle = LoadLibrary(LibPath);
  auto func = LoadFunc(handle, RunFuncName);

  RunGraphFunc(func, params);
  if (!OutputName.empty()) {
    SaveFloatTensor(info, params, OutputName);
  }

  if (RerunTimes != 0) {
    std::chrono::duration<double> elapsed(0);
    for (int i = 0; i < RerunTimes; ++i) {
      ClearOutputTensor(info, params);
      auto start = std::chrono::high_resolution_clock::now();
      RunGraphFunc(func, params);
      auto end = std::chrono::high_resolution_clock::now();
      elapsed += (end - start);
    }
    llvm::outs() << "Run \t" << RerunTimes << "\ttimes\n";
    llvm::outs() << "Average time:\t" << elapsed.count() / RerunTimes
                 << "\tseconds\n";
  }

  dlclose(handle);

  return 0;
}