#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Operation.h"

#include "multi-device-model-compiler/Dialect/Device/IR/Device.h"
#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h"

using namespace mlir;
using namespace multi_device;

namespace multi_device {
namespace device {
#define GEN_PASS_DEF_ADDDEVICETYPETOFUNC
#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h.inc"
} // namespace device
} // namespace multi_device

namespace {
struct AddDeviceTypeToFuncPass final
    : public multi_device::device::impl::AddDeviceTypeToFuncBase<
          AddDeviceTypeToFuncPass> {
  using AddDeviceTypeToFuncBase::AddDeviceTypeToFuncBase;

  AddDeviceTypeToFuncPass(const device::AddDeviceTypeToFuncOptions &options) {
    this->deviceType = options.deviceType;
  }
  void runOnOperation() override;
};
} // namespace

void AddDeviceTypeToFuncPass::runOnOperation() {
  ModuleOp op = getOperation();
  op->setAttr("module.device",
              device::DeviceTypeAttr::get(op.getContext(), deviceType));
}

std::unique_ptr<OperationPass<ModuleOp>>
multi_device::device::createAddDeviceTypeToFuncPass() {
  return std::make_unique<AddDeviceTypeToFuncPass>();
}

std::unique_ptr<mlir::OperationPass<ModuleOp>>
multi_device::device::createAddDeviceTypeToFuncPass(
    const device::AddDeviceTypeToFuncOptions &options) {
  return std::make_unique<AddDeviceTypeToFuncPass>(options);
}