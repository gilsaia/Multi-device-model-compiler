#ifndef MULTI_DEVICE_MODEL_COMPILER_TOOLS_MULTI_DEVICE_LIB_GENERATE_H_
#define MULTI_DEVICE_MODEL_COMPILER_TOOLS_MULTI_DEVICE_LIB_GENERATE_H_
#include <string>
#include <vector>

#include "mlir/Support/LLVM.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"

namespace multi_device {
struct Command {

  std::string _path;
  std::vector<std::string> _args;

  Command(std::string exePath)
      : _path(std::move(exePath)),
        _args({llvm::sys::path::filename(_path).str()}) {}

  Command &appendStr(const std::string &arg);
  Command &appendStrOpt(const std::optional<std::string> &arg);
  Command &appendList(const std::vector<std::string> &args);
  Command &resetArgs();
  int exec(std::string wdir = "") const;
};

int OptLLVMIR(std::string inputNameWithExt, std::string optNameWithExt);
int GenObjectFromLLVMIR(std::string inputNameWithExt,
                        std::string ObjectNameWithExt);
int GenLibraryFromObject(std::string inputNameWithExt,
                         std::string LibraryNameWithExt);
int GenLLFromBC(std::string inputNameWithExt, std::string LLNameWithExt);

} // namespace multi_device

#endif