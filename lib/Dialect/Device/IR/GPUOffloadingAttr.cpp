#include "multi-device-model-compiler/Dialect/Device/IR/Device.h"

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace multi_device;
using namespace multi_device::device;

mlir::LogicalResult device::GPUOffloadingAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::llvm::StringRef kernel, Attribute target) {
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
  assert(binaryOp && "The binary operation must be non null.");
  if (!binaryOp) {
    return failure();
  }

  auto op = mlir::dyn_cast<gpu::BinaryOp>(binaryOp);
  if (!op) {
    binaryOp->emitError("Operation must be a GPU binary.");
    return failure();
  }

  llvm::ArrayRef<mlir::Attribute> objects = op.getObjectsAttr().getValue();

  int64_t index = -1;
  if (Attribute target = getTarget()) {
    // If the target attribute is a number it is the index. Otherwise compare
    // the attribute to every target inside the object array to find the index.
    if (auto indexAttr = mlir::dyn_cast<IntegerAttr>(target)) {
      index = indexAttr.getInt();
    } else {
      for (auto [i, attr] : llvm::enumerate(objects)) {
        auto obj = mlir::dyn_cast<gpu::ObjectAttr>(attr);
        if (obj.getTarget() == target) {
          index = i;
        }
      }
    }
  } else {
    // If the target attribute is null then it's selecting the first object in
    // the object array.
    index = 0;
  }

  if (index < 0 || index >= static_cast<int64_t>(objects.size())) {
    op->emitError("The requested target object couldn't be found.");
    return failure();
  }
  auto object = mlir::dyn_cast<gpu::ObjectAttr>(objects[index]);

  llvm::StringRef fileName = getKernel().str() + ".bc";
  std::error_code ec;
  llvm::raw_fd_ostream fs(fileName, ec);
  if (ec) {
    op->emitError("The requested file can't create.");
    return failure();
  }
  fs << object.getObject().getValue();
  fs.close();

  return success();
}
LogicalResult device::GPUOffloadingAttr::launchKernel(
    Operation *launchFunc, Operation *binaryOp,
    llvm::IRBuilderBase &hostBuilder,
    LLVM::ModuleTranslation &hostModuleTranslation) const {
  return success();
}
