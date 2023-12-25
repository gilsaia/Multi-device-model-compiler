#ifndef MULTI_DEVICE_MODEL_COMPILER_UTILS_COMPILEOPTIONS_H_
#define MULTI_DEVICE_MODEL_COMPILER_UTILS_COMPILEOPTIONS_H_

#include "llvm/Support/CommandLine.h"

namespace multi_device {
/* Compiler optimization level (traditional -O0 ... -O3 flags) */
typedef enum { O0 = 0, O1, O2, O3 } OptLevel;

typedef enum { CPU = 0, GPU, TPU } TargetDevice;

extern llvm::cl::OptionCategory MultiDeviceCompileOptions;
extern llvm::cl::OptionCategory MultiDeviceLibGenOptions;

extern llvm::cl::opt<bool> preserveOptIR;
extern llvm::cl::opt<bool> preserveObject;
extern llvm::cl::opt<std::string> mtriple;
extern llvm::cl::opt<std::string> mcpu;
extern llvm::cl::opt<std::string> march;
extern llvm::cl::opt<OptLevel> OptimizationLevel;
extern llvm::cl::opt<TargetDevice> Device;

std::string getTargetTriple();
std::string getTargetArch();
std::string getTargetCPU();
std::string getOptimizationLevel();
std::string getLibraryExt();

} // namespace multi_device

#endif