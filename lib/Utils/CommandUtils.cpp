#include "multi-device-model-compiler/Utils/CommandUtils.h"
#include "multi-device-model-compiler/Utils/CompileOptions.h"

#include "ExternalUtil.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"

namespace multi_device {

// Append a single string argument.
Command &Command::appendStr(const std::string &arg) {
  if (arg.size() > 0)
    _args.emplace_back(arg);
  return *this;
}

// Append a single optional string argument.
Command &Command::appendStrOpt(const llvm::Optional<std::string> &arg) {
  if (arg.has_value())
    _args.emplace_back(arg.value());
  return *this;
}

// Append a list of string arguments.
Command &Command::appendList(const std::vector<std::string> &args) {
  _args.insert(_args.end(), args.begin(), args.end());
  return *this;
}

// Reset arguments.
Command &Command::resetArgs() {
  auto exeFileName = _args.front();
  _args.clear();
  _args.emplace_back(exeFileName);
  return *this;
}

// Execute command in current work directory.
//
// If the optional wdir is specified, the command will be executed
// in the specified work directory. Current work directory is
// restored after the command is executed.
//
// Return 0 on success, error value otherwise.
int Command::exec(std::string wdir) const {
  auto argsRef = std::vector<llvm::StringRef>(_args.begin(), _args.end());

  // If a work directory is specified, save the current work directory
  // and switch into it. Note that if wdir is empty, new_wdir will be
  // cur_wdir.
  llvm::SmallString<8> cur_wdir;
  llvm::SmallString<8> new_wdir(wdir);
  llvm::sys::fs::current_path(cur_wdir);
  llvm::sys::fs::make_absolute(cur_wdir, new_wdir);
  std::error_code ec = llvm::sys::fs::set_current_path(new_wdir);
  if (ec.value()) {
    llvm::errs() << llvm::StringRef(new_wdir).str() << ": " << ec.message()
                 << "\n";
    return ec.value();
  }

  std::string errMsg;
  int rc = llvm::sys::ExecuteAndWait(
      _path, llvm::ArrayRef(argsRef),
      /*Env=*/std::nullopt, /*Redirects=*/std::nullopt,
      /*SecondsToWait=*/0, /*MemoryLimit=*/0, &errMsg);

  if (rc != 0) {
    llvm::errs() << llvm::join(argsRef, " ") << "\n"
                 << "Error message: " << errMsg << "\n"
                 << "Program path: " << _path << "\n"
                 << "Command execution failed."
                 << "\n";
    return rc;
  }

  // Restore saved work directory.
  llvm::sys::fs::set_current_path(cur_wdir);
  return 0;
}

int OptLLVMIR(std::string inputNameWithExt, std::string optNameWithExt) {
  Command optCommand(kOptPath);
  int rc = optCommand.appendStr(getOptimizationLevel())
               .appendStr(getTargetTriple())
               .appendStr(getTargetArch())
               .appendStr(getTargetCPU())
               .appendStr("-S")
               .appendList({"-o", optNameWithExt})
               .appendStr(inputNameWithExt)
               .exec();
  return rc;
}

int GenObjectFromLLVMIR(std::string inputNameWithExt,
                        std::string ObjectNameWithExt) {
  Command objCommand(kLlcPath);
  int rc = objCommand.appendStr(getOptimizationLevel())
               .appendStr(getTargetTriple())
               .appendStr(getTargetArch())
               .appendStr(getTargetCPU())
               .appendStr("-filetype=obj")
               .appendStr("-relocation-model=pic")
               .appendList({"-o", ObjectNameWithExt})
               .appendStr(inputNameWithExt)
               .exec();
  return rc;
}

int GenLibraryFromObject(std::string inputNameWithExt,
                         std::string LibraryNameWithExt) {
  Command linkCommand(kCxxPath);
  int rc = linkCommand.appendStr(inputNameWithExt)
               .appendList({"-shared", "-fPIC"})
               .appendList({"-o", LibraryNameWithExt})
               .exec();
  return rc;
}
} // namespace multi_device
