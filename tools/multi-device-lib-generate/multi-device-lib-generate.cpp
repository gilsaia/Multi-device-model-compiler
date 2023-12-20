#include "multi-device-model-compiler/Utils/CommandUtils.h"
#include "multi-device-model-compiler/Utils/CompileOptions.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"

#include <regex>

static llvm::cl::opt<std::string>
    inputFileWithExt(llvm::cl::Positional, llvm::cl::desc("<input file>"),
                     llvm::cl::init("-"),
                     llvm::cl::cat(multi_device::MultiDeviceLibGenOptions));

static llvm::cl::opt<std::string>
    outputFileName("o", llvm::cl::desc("Output filename"),
                   llvm::cl::ValueRequired,
                   llvm::cl::cat(multi_device::MultiDeviceLibGenOptions));

using namespace multi_device;

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::HideUnrelatedOptions(
      {&MultiDeviceCompileOptions, &MultiDeviceLibGenOptions});

  if (!llvm::cl::ParseCommandLineOptions(argc, argv,
                                         "Multi device library generater\n")) {
    llvm::errs() << "Failed to parse options\n";
    return 1;
  }

  std::string inputFileWithoutExt =
      inputFileWithExt.substr(0, inputFileWithExt.find_last_of('.'));

  bool b = false;
  if (outputFileName == "" ||
      (b = std::regex_match(
           outputFileName.substr(outputFileName.find_last_of("/\\") + 1),
           std::regex("[\\.]*$")))) {
    if (b)
      llvm::errs() << "Invalid -o option value " << outputFileName
                   << " ignored.\n";
    outputFileName =
        inputFileWithExt.substr(0, inputFileWithExt.find_last_of("."));
  }

  int rc;
  std::string optFileWithExt = inputFileWithoutExt + "_opt.ll";
  rc = OptLLVMIR(inputFileWithExt, optFileWithExt);
  if (rc) {
    llvm::errs() << "Opt LLVM IR wrong.\n";
    return rc;
  }
  llvm::FileRemover optRemover(optFileWithExt, !preserveOptIR);

  std::string objectFileWithExt = inputFileWithoutExt + ".o";
  rc = GenObjectFromLLVMIR(optFileWithExt, objectFileWithExt);
  if (rc) {
    llvm::errs() << "Gen object file wrong.\n";
    return rc;
  }
  llvm::FileRemover objectRemover(objectFileWithExt, !preserveObject);

  std::string outputFileWithExt = outputFileName + getLibraryExt();
  rc = GenLibraryFromObject(objectFileWithExt, outputFileWithExt);
  if (rc) {
    llvm::errs() << "Gen library wrong.\n";
    return rc;
  }

  return 0;
}