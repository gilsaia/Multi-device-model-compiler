#include "multi-device-model-compiler/Runtime/CUDA/CudaRuntimeWrappers.h"
#include "multi-device-model-compiler/Runtime/ModelInfo.h"
#include "multi-device-model-compiler/Runtime/RuntimeUtil.h"
#include "multi-device-model-compiler/Runtime/TensorDescripter.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

#include <chrono>
#include <dlfcn.h>
#include <string>

typedef void (*RunGraph)(Float3DTensor *, Float3DTensor *, Float3DTensor *);

llvm::cl::OptionCategory
    MultiDeviceGpuRunnerOptions("Options use for gpu runner", "");

llvm::cl::opt<std::string> LibPath(llvm::cl::value_desc("Model Library Path"),
                                   llvm::cl::desc("<lib>"), llvm::cl::Required,
                                   llvm::cl::Positional,
                                   llvm::cl::cat(MultiDeviceGpuRunnerOptions));
llvm::cl::opt<std::string>
    RunFuncName("func", llvm::cl::desc("Model Run Func Name"),
                llvm::cl::ValueRequired,
                llvm::cl::init("_mlir_ciface_main_graph"),
                llvm::cl::cat(MultiDeviceGpuRunnerOptions));

llvm::cl::opt<std::string>
    OutputName("out-data", llvm::cl::desc("Output file name"),
               llvm::cl::ValueRequired, llvm::cl::init(""),
               llvm::cl::cat(MultiDeviceGpuRunnerOptions));

llvm::cl::opt<int> RerunTimes("rerun", llvm::cl::desc("Rerun time"),
                              llvm::cl::init(0), llvm::cl::ValueRequired,
                              llvm::cl::cat(MultiDeviceGpuRunnerOptions));

llvm::cl::opt<std::string>
    OpsLibPath("ops", llvm::cl::desc("Ops library path"),
               llvm::cl::init("./ops.cubin"), llvm::cl::ValueRequired,
               llvm::cl::cat(MultiDeviceGpuRunnerOptions));

using namespace multi_device;

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::HideUnrelatedOptions(
      {&MultiDeviceGpuRunnerOptions, &MultiDeviceModelOptions});

  if (!llvm::cl::ParseCommandLineOptions(argc, argv,
                                         "Multi device gpu runner\n")) {
    llvm::errs() << "Failed to parse options\n";
    return 1;
  }

  mgpuModuleFileLoad(OpsLibPath.c_str());

  ModelInfo *info = ModelInfo::ParseModelInfo();
  auto params = GetParamsVec(info);

  void *handle = LoadLibrary(LibPath);
  auto func = LoadFunc(handle, RunFuncName);

  RunGraphFunc(func, params);
  if (!OutputName.empty()) {
    SaveFloatTensor(info, params, OutputName);
  }
  auto stream = mgpuStreamCreate();
  if (RerunTimes != 0) {
    auto eventBegin = mgpuEventEnableTimeCreate(),
         eventEnd = mgpuEventEnableTimeCreate();
    float elapsed = 0;
    for (int i = 0; i < RerunTimes; ++i) {
      ClearOutputTensor(info, params);
      mgpuEventRecord(eventBegin, stream);
      RunGraphFunc(func, params);
      mgpuEventRecord(eventEnd, stream);
      mgpuEventSynchronize(eventEnd);
      elapsed += mgpuEventElapsedTime(eventBegin, eventEnd);
    }
    mgpuEventDestroy(eventBegin);
    mgpuEventDestroy(eventEnd);
    llvm::outs() << "Run \t" << RerunTimes << "\ttimes\n";
    llvm::outs() << "Average time:\t" << elapsed / (RerunTimes * 1000)
                 << "\tseconds\n";
  }

  mgpuModuleUnload();
  dlclose(handle);

  return 0;
}