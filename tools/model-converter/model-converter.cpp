#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"

#include "src/Builder/FrontendDialectTransformer.hpp"
#include "src/Compiler/CompilerUtils.hpp"

#include "multi-device-model-compiler/Utils/CompileUtils.h"

llvm::cl::OptionCategory ModelConverterOptions("Converter Options", "");

int main(int argc, char **argv) {
  multi_device::removeUnrelatedOptions({&ModelConverterOptions});

  llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::Required,
      llvm::cl::cat(ModelConverterOptions));
  llvm::cl::opt<std::string> outputBaseName(
      "o",
      llvm::cl::desc("Base path for output files, extensions will be added."),
      llvm::cl::value_desc("path"), llvm::cl::ValueRequired,
      llvm::cl::cat(ModelConverterOptions));

  llvm::cl::HideUnrelatedOptions({&ModelConverterOptions});

  if (!llvm::cl::ParseCommandLineOptions(argc, argv,
                                         "Onnx model converter\n")) {
    llvm::errs() << "Failed to parse options\n";
    return 1;
  }

  mlir::MLIRContext context;
  onnx_mlir::loadDialects(context);

  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string errorMessage;
  int rc = onnx_mlir::processInputFile(inputFilename, context, module,
                                       &errorMessage);
  if (rc != 0) {
    if (!errorMessage.empty()) {
      llvm::errs() << errorMessage << "\n";
      return 1;
    }
  }

  if (outputBaseName == "") {
    outputBaseName = inputFilename.substr(0, inputFilename.find_last_of("."));
  }

  std::string outputNameWithExt = outputBaseName + ".onnx.mlir";
  std::string outputTmpWithExt = outputBaseName + ".tmp";
  onnx_mlir::outputCode(module, outputNameWithExt);
  // Elide element attributes if larger than 100
  onnx_mlir::outputCode(module, outputTmpWithExt, 100);

  return 0;
}