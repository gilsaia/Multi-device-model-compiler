#include "mlir/IR/Operation.h"

#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h"

using namespace mlir;

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
  void runOnOperation() override;
};
} // namespace

void AddDeviceTypeToFuncPass::runOnOperation() {}

std::unique_ptr<OperationPass<func::FuncOp>>
multi_device::device::createAddDeviceTypeToFuncPass() {
  return std::make_unique<AddDeviceTypeToFuncPass>();
}