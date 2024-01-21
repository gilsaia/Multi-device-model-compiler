#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"

using namespace mlir;

namespace multi_device {
namespace device {
#define GEN_PASS_DEF_ELIMINATEGPUMEMREFSPACE
#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h.inc"
} // namespace device
} // namespace multi_device

namespace {
class EliminateGPUMemrefSpacePass final
    : public multi_device::device::impl::EliminateGPUMemrefSpaceBase<
          EliminateGPUMemrefSpacePass> {
  using EliminateGPUMemrefSpaceBase::EliminateGPUMemrefSpaceBase;

  void runOnOperation() override final;
};
} // namespace

void EliminateGPUMemrefSpacePass::runOnOperation() {
  auto moduleOp = getOperation();
  moduleOp.walk([](gpu::AllocOp op) {
    auto srcMemrefType = op.getMemref().getType();
    auto dstMemrefType = MemRefType::get(srcMemrefType.getShape(),
                                         srcMemrefType.getElementType(),
                                         srcMemrefType.getLayout());
    op.getMemref().setType(dstMemrefType);
    return WalkResult::advance();
  });
}