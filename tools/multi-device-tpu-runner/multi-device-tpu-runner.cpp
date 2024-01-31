#include "multi-device-model-compiler/Runtime/ModelInfo.h"
#include "multi-device-model-compiler/Runtime/RuntimeUtil.h"
#include "multi-device-model-compiler/Runtime/TPU/TPURuntimeWrappers.h"
#include "multi-device-model-compiler/Runtime/TensorDescripter.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

#include <chrono>
#include <string>

llvm::cl::OptionCategory
    MultiDeviceTpuRunnerOptions("Options use for tpu runner", "");

llvm::cl::opt<std::string> LibPath(llvm::cl::value_desc("Model Library Path"),
                                   llvm::cl::desc("<lib>"), llvm::cl::Required,
                                   llvm::cl::Positional,
                                   llvm::cl::cat(MultiDeviceTpuRunnerOptions));
llvm::cl::opt<std::string>
    RunFuncName("func", llvm::cl::desc("Model Run Func Name"),
                llvm::cl::ValueRequired,
                llvm::cl::init("_mlir_ciface_main_graph"),
                llvm::cl::cat(MultiDeviceTpuRunnerOptions));

llvm::cl::opt<std::string>
    OutputName("out-data", llvm::cl::desc("Output file name"),
               llvm::cl::ValueRequired, llvm::cl::init(""),
               llvm::cl::cat(MultiDeviceTpuRunnerOptions));

llvm::cl::opt<int> RerunTimes("rerun", llvm::cl::desc("Rerun time"),
                              llvm::cl::init(0), llvm::cl::ValueRequired,
                              llvm::cl::cat(MultiDeviceTpuRunnerOptions));

using namespace multi_device;

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::HideUnrelatedOptions(
      {&MultiDeviceTpuRunnerOptions, &MultiDeviceModelOptions});

  if (!llvm::cl::ParseCommandLineOptions(argc, argv,
                                         "Multi device tpu runner\n")) {
    llvm::errs() << "Failed to parse options\n";
    return 1;
  }

  ModelInfo *info = ModelInfo::ParseModelInfo();

  llvm::errs() << "Before init\n";
  auto handle = InitHandle();
  auto runtime = InitRuntime(handle);

  llvm::errs() << "After init\n";

  if (!LoadBModel(runtime, LibPath)) {
    llvm::errs() << "Failed to load bmodel\n";
    return 1;
  }

  auto tensors = GetTensorData(info);
  auto name = GetNetName(runtime);

  LaunchModel(info, runtime, tensors, name);
  if (!OutputName.empty()) {
    SaveTensorData(info, tensors, OutputName);
  }

  if (RerunTimes != 0) {
    std::chrono::duration<double> elapsed(0);
    for (int i = 0; i < RerunTimes; ++i) {
      auto start = std::chrono::high_resolution_clock::now();
      LaunchModel(info, runtime, tensors, name);
      auto end = std::chrono::high_resolution_clock::now();
      elapsed += (end - start);
    }
    llvm::outs() << "Run \t" << RerunTimes << "\ttimes\n";
    llvm::outs() << "Average time:\t" << elapsed.count() / RerunTimes
                 << "\tseconds\n";
  }

  DestroyRuntime(handle, runtime);

  return 0;
}