#include "multi-device-model-compiler/Dialect/Device/IR/Device.h"
#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/OneToNTypeConversion.h"

using namespace mlir;

namespace multi_device {
namespace device {
#define GEN_PASS_DEF_ASYNCDEPENDENCYCONVERT
#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h.inc"
} // namespace device
} // namespace multi_device

namespace {
class AsyncDependencyConvertPass final
    : public multi_device::device::impl::AsyncDependencyConvertBase<
          AsyncDependencyConvertPass> {
  using AsyncDependencyConvertBase::AsyncDependencyConvertBase;
  void runOnOperation() override final;
};
} // namespace

namespace {
void changeAsyncDependencies(MutableOperandRange range, Value val,
                             Operation *op, OneToNPatternRewriter &rewriter) {
  llvm::SmallVector<Value, 4> needDependencies;
  for (auto dependency : range) {
    if (!mlir::isa<gpu::WaitOp>(dependency.getDefiningOp())) {
      needDependencies.push_back(dependency);
    }
  }
  range.clear();
  range.append(val);
  rewriter.setInsertionPoint(op);
  for (auto &src : needDependencies) {
    rewriter.create<multi_device::device::RecordOp>(op->getLoc(), src, val);
  }
}
} // namespace

void AsyncDependencyConvertPass::runOnOperation() {
  auto funcOp = getOperation();
  OneToNPatternRewriter rewriter(&getContext());
  rewriter.setInsertionPointToStart(&funcOp.front());
  auto dataStream = rewriter.create<multi_device::device::WaitOp>(
      rewriter.getInsertionPoint()->getLoc(),
      rewriter.getType<gpu::AsyncTokenType>(), ValueRange{});
  dataStream.setName(rewriter.getStringAttr("DataStream"));
  auto kernelStream = rewriter.create<multi_device::device::WaitOp>(
      rewriter.getInsertionPoint()->getLoc(),
      rewriter.getType<gpu::AsyncTokenType>(), ValueRange{});
  kernelStream.setName(rewriter.getStringAttr("KernelStream"));

  funcOp.walk([&](gpu::AsyncOpInterface op) {
    rewriter.setInsertionPointAfter(op);
    if (mlir::isa<multi_device::device::WaitOp>(op)) {
      return WalkResult::advance();
    }
    if (mlir::isa<gpu::WaitOp>(op)) {
      auto wait = mlir::cast<gpu::WaitOp>(op);
      wait.getAsyncDependenciesMutable().clear();
      return WalkResult::advance();
    }
    if (mlir::isa<gpu::AllocOp>(op)) {
      auto alloc = mlir::cast<gpu::AllocOp>(op);
      ::changeAsyncDependencies(alloc.getAsyncDependenciesMutable(),
                                dataStream.getAsyncToken(), op, rewriter);
      return WalkResult::advance();
    }
    if (mlir::isa<gpu::DeallocOp>(op)) {
      auto dealloc = mlir::cast<gpu::DeallocOp>(op);
      ::changeAsyncDependencies(dealloc.getAsyncDependenciesMutable(),
                                dataStream.getAsyncToken(), op, rewriter);
      return WalkResult::advance();
    }
    if (mlir::isa<gpu::MemcpyOp>(op)) {
      auto memcpy = mlir::cast<gpu::MemcpyOp>(op);
      ::changeAsyncDependencies(memcpy.getAsyncDependenciesMutable(),
                                dataStream.getAsyncToken(), op, rewriter);
      return WalkResult::advance();
    }
    if (mlir::isa<gpu::LaunchFuncOp>(op)) {
      auto launch = mlir::cast<gpu::LaunchFuncOp>(op);
      ::changeAsyncDependencies(launch.getAsyncDependenciesMutable(),
                                kernelStream.getAsyncToken(), op, rewriter);
      rewriter.create<multi_device::device::RecordOp>(
          op.getLoc(), dataStream.getAsyncToken(),
          kernelStream.getAsyncToken());
      return WalkResult::advance();
    }
    if (mlir::isa<multi_device::device::MatmulOp>(op)) {
      auto matmul = mlir::cast<multi_device::device::MatmulOp>(op);
      ::changeAsyncDependencies(matmul.getAsyncDependenciesMutable(),
                                kernelStream.getAsyncToken(), op, rewriter);
      rewriter.create<multi_device::device::RecordOp>(
          op.getLoc(), dataStream.getAsyncToken(),
          kernelStream.getAsyncToken());
      return WalkResult::advance();
    }
    if (mlir::isa<multi_device::device::Conv2DOp>(op)) {
      auto conv2d = cast<multi_device::device::Conv2DOp>(op);
      ::changeAsyncDependencies(conv2d.getAsyncDependenciesMutable(),
                                kernelStream.getAsyncToken(), op, rewriter);
      rewriter.create<multi_device::device::RecordOp>(
          op.getLoc(), dataStream.getAsyncToken(),
          kernelStream.getAsyncToken());
      return WalkResult::advance();
    }
    if (mlir::isa<multi_device::device::Pool2DOp>(op)) {
      auto pool2d = cast<multi_device::device::Pool2DOp>(op);
      ::changeAsyncDependencies(pool2d.getAsyncDependenciesMutable(),
                                kernelStream.getAsyncToken(), op, rewriter);
      rewriter.create<multi_device::device::RecordOp>(
          op.getLoc(), dataStream.getAsyncToken(),
          kernelStream.getAsyncToken());
      return WalkResult::advance();
    }
    return WalkResult::advance();
  });
  rewriter.setInsertionPointAfter(funcOp.back().back().getPrevNode());
  auto streamEnd = rewriter.create<multi_device::device::WaitOp>(
      rewriter.getInsertionPoint()->getLoc(), TypeRange{},
      ValueRange{dataStream.getAsyncToken(), kernelStream.getAsyncToken()});
  streamEnd.setName("StreamEnd");
}