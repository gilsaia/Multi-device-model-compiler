#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"

using namespace mlir;

namespace multi_device {
namespace device {
#define GEN_PASS_DEF_COMBINEGPUKERNEL
#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h.inc"
} // namespace device
} // namespace multi_device

namespace {
class CombineGPUKernelPass final
    : public multi_device::device::impl::CombineGPUKernelBase<
          CombineGPUKernelPass> {
  using CombineGPUKernelBase::CombineGPUKernelBase;
  void runOnOperation() override final;
};
} // namespace

void CombineGPUKernelPass::runOnOperation() {
  auto moduleOp = getOperation();
  auto ctx = moduleOp.getContext();
  llvm::SmallVector<gpu::GPUFuncOp> funcs;
  OpBuilder builder(ctx);
  moduleOp.walk([&moduleOp, &ctx, &funcs, &builder](gpu::GPUFuncOp op) {
    auto gpuModule = op.getParentOp();
    op.setName(gpuModule.getName());
    auto cloneGpuOp = op.clone();
    funcs.push_back(cloneGpuOp);
    return WalkResult::advance();
  });
  moduleOp.walk([](gpu::GPUModuleOp op) { op.erase(); });
  auto loc = moduleOp.getBody()->back().getLoc();
  builder.setInsertionPointToEnd(moduleOp.getBody());
  auto newGpuModule = builder.create<gpu::GPUModuleOp>(loc, "Ops");
  builder.setInsertionPointToStart(newGpuModule.getBody());
  for (auto gpuFunc : funcs) {
    builder.insert(gpuFunc);
  }
  moduleOp.walk([&builder](gpu::LaunchFuncOp launch) {
    auto kernelNameAttr = FlatSymbolRefAttr::get(
        builder.getStringAttr(launch.getKernelModuleName().getValue()));
    auto newKernelAttr =
        SymbolRefAttr::get(builder.getStringAttr("Ops"), {kernelNameAttr});
    launch.setKernelAttr(newKernelAttr);
    return WalkResult::advance();
  });
}