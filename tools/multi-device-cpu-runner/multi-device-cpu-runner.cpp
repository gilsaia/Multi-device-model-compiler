#include "multi-device-model-compiler/Runtime/ModelInfo.h"
#include "multi-device-model-compiler/Runtime/TensorDescripter.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

#include <dlfcn.h>
#include <string>

typedef void (*RunGraph)(Float3DTensor *, Float3DTensor *, Float3DTensor *);

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

  void *handle = dlopen(LibPath.c_str(), RTLD_LAZY);
  if (!handle) {
    llvm::errs() << "Failed to open library :" << dlerror() << "\n";
    return 1;
  }

  RunGraph func = (RunGraph)dlsym(handle, RunFuncName.c_str());

  const char *err = dlerror();
  if (err) {
    llvm::errs() << "Failed to find run function: " << err << "\n";
    return 1;
  }

  ModelInfo *info = ModelInfo::ParseModelInfo();

  Float3DTensor *a = Float3DTensor::CreateTensor(info->GetInputSize(0)),
                *b = Float3DTensor::CreateTensor(info->GetInputSize(1)),
                *c = Float3DTensor::CreateTensor(info->GetOutputSize(0));
  a->InitData(), b->InitData();
  func(c, a, b);

  dlclose(handle);

  return 0;
}