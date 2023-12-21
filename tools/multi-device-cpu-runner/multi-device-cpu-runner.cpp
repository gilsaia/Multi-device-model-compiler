#include "multi-device-model-compiler/Runtime/TensorDescripter.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

#include <dlfcn.h>
#include <string>

typedef void (*RunGraph)(Float3DTensor *, Float3DTensor *, Float3DTensor *);

llvm::cl::opt<std::string> LibPath(llvm::cl::desc("Model Library Path"),
                                   llvm::cl::Required, llvm::cl::Positional);

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

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

  RunGraph func = (RunGraph)dlsym(handle, "_mlir_ciface_main_graph");

  const char *err = dlerror();
  if (err) {
    llvm::errs() << "Failed to find run function: " << err << "\n";
    return 1;
  }
  std::vector<int> sizes{3, 640, 640}, strides{1, 1, 1};
  Float3DTensor *a = Float3DTensor::CreateTensor(sizes, strides),
                *b = Float3DTensor::CreateTensor(sizes, strides),
                *c = Float3DTensor::CreateTensor(sizes, strides);
  func(c, a, b);

  dlclose(handle);

  return 0;
}