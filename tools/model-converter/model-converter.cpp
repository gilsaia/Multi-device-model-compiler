#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"

#include "Builder/FrontendDialectTransformer.hpp"

int main(int argc, char **argv) {
  llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::Required);
  llvm::cl::opt<std::string> outputBaseName(
      "o",
      llvm::cl::desc("Base path for output files, extensions will be added."),
      llvm::cl::value_desc("path"), llvm::cl::ValueRequired);
  if (!llvm::cl::ParseCommandLineOptions(argc, argv,
                                         "Onnx model converter\n")) {
    llvm::errs() << "Failed to parse options\n";
    return 1;
  }

  return 0;
  // mlir::DialectRegistry registry;
  // return mlir::asMainReturnCode(
  //     mlir::MlirOptMain(argc, argv, "Model-Converter", registry));
}