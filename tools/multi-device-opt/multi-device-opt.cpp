#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Dialect/ONNX/ONNXDialect.hpp"
#include "Pass/Passes.hpp"

#include "multi-device-model-compiler/InitPasses.h"

static llvm::cl::OptionCategory
    MultiDeviceOptOptions("Multi Device OPT Options",
                          "These are opt frontend options.");

static llvm::cl::opt<std::string>
    input_filename(llvm::cl::Positional, llvm::cl::desc("<input file>"),
                   llvm::cl::init("-"), llvm::cl::cat(MultiDeviceOptOptions));

static llvm::cl::opt<std::string>
    output_filename("o", llvm::cl::desc("Output filename"),
                    llvm::cl::value_desc("filename"), llvm::cl::init("-"),
                    llvm::cl::cat(MultiDeviceOptOptions));

using namespace mlir;

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  mlir::DialectRegistry registry;
  registry.insert<mlir::ONNXDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::tosa::TosaDialect>();
  registry.insert<mlir::tensor::TensorDialect>();

  multi_device::initONNXPasses();

  llvm::cl::HideUnrelatedOptions({&MultiDeviceOptOptions});

  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();

  MlirOptMainConfig::registerCLOptions(registry);

  if (!llvm::cl::ParseCommandLineOptions(argc, argv,
                                         "Multi device converter\n")) {
    llvm::errs() << "Failed to parse options\n";
    return 1;
  }

  // Set up the input file.
  std::string error_message;
  auto file = mlir::openInputFile(input_filename, &error_message);
  if (!error_message.empty()) {
    llvm::errs() << "Failure to open file; " << error_message << "\n";
    return failed(LogicalResult::failure());
  }

  auto output = mlir::openOutputFile(output_filename, &error_message);
  if (!error_message.empty()) {
    llvm::errs() << "Failure to compile file; " << error_message << "\n";
    return failed(LogicalResult::failure());
  }

  auto config = mlir::MlirOptMainConfig::createFromCLOptions();

  if (failed(
          mlir::MlirOptMain(output->os(), std::move(file), registry, config))) {
    return mlir::asMainReturnCode(failure());
  }

  output->keep();
  return mlir::asMainReturnCode(success());
}