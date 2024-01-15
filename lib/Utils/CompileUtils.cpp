#include "multi-device-model-compiler/Utils/CompileUtils.h"

namespace multi_device {
void removeUnrelatedOptions(
    const std::vector<llvm::cl::OptionCategory *> Categories) {
  llvm::cl::HideUnrelatedOptions(Categories);

  llvm::StringMap<llvm::cl::Option *> &optMap =
      llvm::cl::getRegisteredOptions();
  for (auto n = optMap.begin(); n != optMap.end(); n++) {
    llvm::cl::Option *opt = n->getValue();
    if (opt->getOptionHiddenFlag() == llvm::cl::ReallyHidden)
      opt->removeArgument();
  }
}
} // namespace multi_device