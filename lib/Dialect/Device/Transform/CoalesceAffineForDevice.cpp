#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

namespace multi_device {
namespace device {
#define GEN_PASS_DEF_COALESCEAFFINEFORDEVICE
#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h.inc"
} // namespace device
} // namespace multi_device

namespace {
class CoalesceAffineForDevicePass final
    : public multi_device::device::impl::CoalesceAffineForDeviceBase<
          CoalesceAffineForDevicePass> {
public:
  using CoalesceAffineForDeviceBase::CoalesceAffineForDeviceBase;
  void runOnOperation() override final;
};
} // namespace

static void CoalesceAffineForOp(affine::AffineForOp loop) {
  llvm::SmallVector<affine::AffineForOp> loops;
  mlir::affine::getPerfectlyNestedLoops(loops, loop);
  if (loops.size() < 2) {
    return;
  }
  int64_t rank = 2;
  loop.walk([&](affine::AffineLoadOp load) {
    rank =
        std::max(rank, load.getMemRefType().cast<mlir::ShapedType>().getRank());
  });
  for (unsigned end = rank; end > 0; --end) {
    unsigned start = 0;
    auto band = llvm::MutableArrayRef(loops.data() + start, end - start);
    if (succeeded(affine::coalesceLoops(band))) {
      return;
    }
  }
}

void CoalesceAffineForDevicePass::runOnOperation() {
  auto moduleOp = getOperation();
  moduleOp.walk([&](affine::AffineForOp loop) {
    if (!mlir::isa<func::FuncOp>(loop->getParentOp())) {
      return WalkResult::advance();
    }
    CoalesceAffineForOp(loop);
    return WalkResult::advance();
  });
}