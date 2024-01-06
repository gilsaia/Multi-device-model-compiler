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

  AddDeviceTypeToFuncPass(const device::AddDeviceTypeToFuncOptions &options)
      : AddDeviceTypeToFuncBase(options) {}
  void runOnOperation() override;
};
} // namespace

void AddDeviceTypeToFuncPass::runOnOperation() {
  ModuleOp op = getOperation();
  op->setAttr("module.device",
              device::DeviceTypeAttr::get(op.getContext(), deviceType));
  if (deviceType == device::DeviceType::TPU) {
    op.setName("main_graph");
    op->setAttr("module.chip", StringAttr::get(op.getContext(), "bm1684x"));
    op->setAttr("module.mode", StringAttr::get(op.getContext(), "F32"));
    op->setAttr("module.asymmetric", BoolAttr::get(op.getContext(), false));
    op->setAttr("module.state",
                StringAttr::get(op->getContext(), "TPU_LOWERED"));
    FileLineColLoc loc = op.getLoc().dyn_cast<FileLineColLoc>();
    if (!loc) {
      llvm_unreachable("Can't Find file name\n");
    }
    auto npz_name = loc.getFilename().str() + ".npz";
    op->setAttr("module.weight_file",
                StringAttr::get(op->getContext(), npz_name));
    op->setAttr("module.FLOPs",
                IntegerAttr::get(IntegerType::get(op->getContext(), 64), 0));
  }
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