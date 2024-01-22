#include "multi-device-model-compiler/Dialect/Device/IR/Device.h"
#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

using namespace mlir;

namespace multi_device {
namespace device {
#define GEN_PASS_DEF_OFFLOADINGLLVMTOGPU
#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h.inc"
} // namespace device
} // namespace multi_device

namespace {
struct OffloadingLLVMToGPUPass final
    : public multi_device::device::impl::OffloadingLLVMToGPUBase<
          OffloadingLLVMToGPUPass> {
  using OffloadingLLVMToGPUBase::OffloadingLLVMToGPUBase;

  OffloadingLLVMToGPUPass(
      const multi_device::device::OffloadingLLVMToGPUOptions &options)
      : OffloadingLLVMToGPUBase(options) {}
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;
};
} // namespace

void OffloadingLLVMToGPUPass::getDependentDialects(
    DialectRegistry &registry) const {
  registerLLVMDialectTranslation(registry);
  registerGPUDialectTranslation(registry);
  registerNVVMTarget(registry);
}

void OffloadingLLVMToGPUPass::runOnOperation() {
  int targetFormat =
      llvm::StringSwitch<int>(compilationTarget)
          .Cases("offloading", "llvm", gpu::TargetOptions::offload)
          .Cases("assembly", "isa", gpu::TargetOptions::assembly)
          .Cases("binary", "bin", gpu::TargetOptions::binary)
          .Default(-1);
  if (targetFormat == -1)
    getOperation()->emitError() << "Invalid format specified.";
  gpu::TargetOptions targetOptions(
      toolkitPath, linkFiles, cmdOptions,
      static_cast<gpu::TargetOptions::CompilationTarget>(targetFormat));
  if (failed(gpu::transformGpuModulesToBinaries(
          getOperation(),
          multi_device::device::GPUOffloadingAttr::get(&getContext(),
                                                       kernelName, nullptr),
          targetOptions)))
    return signalPassFailure();
}