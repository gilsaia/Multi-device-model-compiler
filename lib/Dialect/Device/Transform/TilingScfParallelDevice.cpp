#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"

using namespace mlir;

namespace multi_device {
namespace device {
#define GEN_PASS_DEF_TILINGSCFPARALLELDEVICE
#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h.inc"
} // namespace device
} // namespace multi_device

namespace {
class TilingScfParallelDevicePass final
    : public multi_device::device::impl::TilingScfParallelDeviceBase<
          TilingScfParallelDevicePass> {
public:
  using TilingScfParallelDeviceBase::TilingScfParallelDeviceBase;
  TilingScfParallelDevicePass(
      const multi_device::device::TilingScfParallelDeviceOptions &options)
      : TilingScfParallelDeviceBase(options) {}
  void runOnOperation() override final;
};
} // namespace

void TilingParallelOp(scf::ParallelOp parallel, int64_t totalTilingSize) {
  auto uppers = parallel.getUpperBound();
  llvm::SmallVector<int64_t, 4> forDims;
  for (auto upper : uppers) {
    auto val = mlir::getConstantIntValue(upper);
    if (val) {
      forDims.push_back(val.value());
    } else {
      forDims.push_back(1);
    }
  }
  llvm::SmallVector<int64_t, 4> tilingDims(forDims.size(), 1);
  int64_t totalTiling = totalTilingSize, remainTiling = totalTiling;
  for (int i = tilingDims.size() - 1; i >= 0; --i) {
    if (forDims[i] > 1 && remainTiling > 1) {
      int64_t tmpTiling = remainTiling;
      while (tmpTiling > forDims[i]) {
        tmpTiling /= 2;
      }
      tilingDims[i] *= tmpTiling;
      remainTiling /= tmpTiling;
    }
  }
  parallel.walk([&](memref::LoadOp op) {
    auto type = op.getMemRefType();
    if (!type.hasRank()) {
      return WalkResult::advance();
    }
    auto shapes = type.getShape();
    if (tilingDims.size() == 1 || tilingDims.size() != shapes.size()) {
      return WalkResult::advance();
    }
    int lastIndex = 0;
    for (int i = tilingDims.size() - 1; i >= lastIndex; --i) {
      if (tilingDims[i] >= shapes[i]) {
        continue;
      }
      while (tilingDims[i] < shapes[i]) {
        if (lastIndex == i) {
          break;
        }
        if (tilingDims[lastIndex] == 1) {
          ++lastIndex;
          continue;
        }
        tilingDims[lastIndex] /= 2;
        tilingDims[i] *= 2;
      }
    }
    return WalkResult::advance();
  });
  tilingDims.back() *= remainTiling;
  scf::tileParallelLoop(parallel, tilingDims, false);
}

void TilingScfParallelDevicePass::runOnOperation() {
  auto moduleOp = getOperation();
  moduleOp.walk([&](scf::ParallelOp parallel) {
    if (mlir::isa<scf::ParallelOp>(parallel->getParentOp())) {
      return WalkResult::advance();
    }
    TilingParallelOp(parallel, totalTilingSize);
    return WalkResult::advance();
  });
}