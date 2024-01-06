#include "multi-device-model-compiler/Utils/CommandUtils.h"
#include "multi-device-model-compiler/Utils/CompileOptions.h"

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
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

int cpuLibGen(std::string &inputFileWithoutExt) {
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

int tpuLibGen() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<tpu::TpuDialect>();
  registry.insert<top::TopDialect>();
  mlir::MLIRContext ctx(registry, mlir::MLIRContext::Threading::DISABLED);
  mlir::ParserConfig config(&ctx);
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(inputFileWithExt, config);
  mlir::PassManager pm(module.get()->getName(),
                       mlir::PassManager::Nesting::Implicit);
  std::string outputFileWithExt = outputFileName + getLibraryExt();
  tpu::CodegenOptions option;
  option.model_file = outputFileWithExt;
  pm.addPass(top::createInitPass());
  pm.addPass(tpu::createCodegenPass(option));
  pm.addPass(top::createDeinitPass());
  if (mlir::failed(pm.run(module.get()))) {
    return 1;
  }
  return 0;
}

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

  switch (Device) {
  case multi_device::TargetDevice::CPU:
    return cpuLibGen(inputFileWithoutExt);
  case multi_device::TargetDevice::TPU:
    return tpuLibGen();
  case multi_device::TargetDevice::GPU:
  default:
    llvm_unreachable("Not Implement Device.");
  }

  return 0;
}