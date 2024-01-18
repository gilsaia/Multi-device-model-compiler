#include "multi-device-model-compiler/Dialect/Device/IR/Device.h"

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace multi_device;
using namespace multi_device::device;

mlir::LogicalResult device::GPUOffloadingAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Attribute target) {
  if (target) {
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(target)) {
      if (intAttr.getInt() < 0) {
        return emitError() << "The object index must be positive.";
      }
    } else if (!(::mlir::isa<mlir::gpu::TargetAttrInterface>(target))) {
      return emitError()
             << "The target attribute must be a GPU Target attribute.";
    }
  }
  return mlir::success();
}

LogicalResult device::GPUOffloadingAttr::embedBinary(
    Operation *binaryOp, llvm::IRBuilderBase &hostBuilder,
    LLVM::ModuleTranslation &hostModuleTranslation) const {
  return success();
}
LogicalResult device::GPUOffloadingAttr::launchKernel(
    Operation *launchFunc, Operation *binaryOp,
    llvm::IRBuilderBase &hostBuilder,
    LLVM::ModuleTranslation &hostModuleTranslation) const {
  return success();
}
