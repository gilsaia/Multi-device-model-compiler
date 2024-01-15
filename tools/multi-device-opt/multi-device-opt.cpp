#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Pass/Passes.hpp"

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

#include "multi-device-model-compiler/Dialect/Device/IR/Device.h"
#include "multi-device-model-compiler/InitUtils.h"
#include "multi-device-model-compiler/Utils/CompileUtils.h"

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

// TPU-MLIR Passes need to set something special,so we need to check if there is
// a `mlir-to-tpu`
bool checkIsTPU(int argc, char **argv) {
  for (int i = argc - 1; i > 0; --i) {
    std::string arg(argv[i]);
    if (arg.find("--mlir-to-tpu") != std::string::npos) {
      return true;
    }
  }
  return false;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  mlir::DialectRegistry registry;
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::vector::VectorDialect>();
  registry.insert<mlir::ONNXDialect>();
  registry.insert<mlir::tosa::TosaDialect>();
  registry.insert<mlir::shape::ShapeDialect>();
  registry.insert<tpu::TpuDialect>();
  registry.insert<multi_device::device::DeviceDialect>();

  mlir::registerTransformsPasses();
  multi_device::initONNXPasses();
  multi_device::initConvertPasses();
  multi_device::initConvertPassPipelines();
  multi_device::initMultiDevicePasses();

  multi_device::removeUnrelatedOptions({&MultiDeviceOptOptions});

  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();

  MlirOptMainConfig::registerCLOptions(registry);

  if (checkIsTPU(argc, argv)) {
    char **n_argv = new char *[argc + 2];
    for (int i = 0; i < argc; ++i) {
      n_argv[i] = argv[i];
    }
    std::string debuginfo("--mlir-print-debuginfo");
    n_argv[argc] = new char[debuginfo.size() + 1];
    strcpy(n_argv[argc], debuginfo.c_str());
    std::string disablethreading("--mlir-disable-threading");
    n_argv[argc + 1] = new char[disablethreading.size() + 1];
    strcpy(n_argv[argc + 1], disablethreading.c_str());
    argc += 2;
    argv = n_argv;
  }

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