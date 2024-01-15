#ifndef MULTI_DEVICE_MODEL_COMPILER_UTILS_COMPILERUTILS_H_
#define MULTI_DEVICE_MODEL_COMPILER_UTILS_COMPILERUTILS_H_

#include "llvm/Support/CommandLine.h"

#include <vector>

namespace multi_device {
void removeUnrelatedOptions(
    const std::vector<llvm::cl::OptionCategory *> Categories);
}

#endif