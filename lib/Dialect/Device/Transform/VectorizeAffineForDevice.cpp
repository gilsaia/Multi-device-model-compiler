#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace multi_device {
namespace device {
#define GEN_PASS_DEF_VECTORIZEAFFINEFORDEVICE
#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h.inc"
} // namespace device
} // namespace multi_device

namespace {
class VectorizeAffineForDevicePass final
    : public multi_device::device::impl::VectorizeAffineForDeviceBase<
          VectorizeAffineForDevicePass> {
public:
  using VectorizeAffineForDeviceBase::VectorizeAffineForDeviceBase;
  VectorizeAffineForDevicePass(
      const multi_device::device::VectorizeAffineForDeviceOptions &options)
      : VectorizeAffineForDeviceBase(options) {}
  void runOnOperation() override final;
};
} // namespace

static void VectorizeAffineForOp(affine::AffineForOp loop,
                                 bool tensorCoreMode) {
  Operation *parentOp = loop->getParentOp();
  llvm::DenseSet<Operation *> parallelLoops;
  parallelLoops.insert(loop);
  unsigned int loopNum = 0;
  int64_t innerStep = 16;
  loop.walk(
      [&parallelLoops, &loopNum, &innerStep](affine::AffineForOp subLoop) {
        if (affine::isLoopParallel(subLoop)) {
          parallelLoops.insert(subLoop);
        }
        auto depth = affine::getNestingDepth(subLoop);
        loopNum = std::max(loopNum, affine::getNestingDepth(subLoop));
        if (depth == loopNum) {
          auto num = subLoop.getConstantUpperBound();
          while (num < innerStep) {
            innerStep /= 2;
          }
        }
      });
  llvm::SmallVector<int64_t> vectorSizes, fastestVaryingPattern;
  if (!tensorCoreMode) {
    vectorSizes.push_back(innerStep);
    fastestVaryingPattern.push_back(0);
  } else {
    llvm_unreachable("Not implement yet");
  }
  affine::vectorizeAffineLoops(parentOp, parallelLoops, vectorSizes,
                               fastestVaryingPattern);
}

void VectorizeAffineForDevicePass::runOnOperation() {
  auto moduleOp = getOperation();
  moduleOp.walk([&](affine::AffineForOp loop) {
    if (!mlir::isa<func::FuncOp>(loop->getParentOp()) ||
        !affine::isLoopParallel(loop)) {
      return WalkResult::advance();
    }
    VectorizeAffineForOp(loop, tensorCoreMode);
    return WalkResult::advance();
  });
}