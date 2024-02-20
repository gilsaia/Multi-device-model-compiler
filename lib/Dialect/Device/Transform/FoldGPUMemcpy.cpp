#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

namespace multi_device {
namespace device {
#define GEN_PASS_DEF_FOLDGPUMEMCPY
#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h.inc"
} // namespace device
} // namespace multi_device

namespace {
class FoldGPUMemcpyPass final
    : public multi_device::device::impl::FoldGPUMemcpyBase<FoldGPUMemcpyPass> {
public:
  using FoldGPUMemcpyBase::FoldGPUMemcpyBase;
  void runOnOperation() override final;
};
} // namespace

static void FoldGPUMemcpydth(gpu::MemcpyOp dth) { dth.erase(); }

static void FoldGPUMemcpyhtd(gpu::MemcpyOp dth, gpu::MemcpyOp htd) {
  auto srcDevice = dth.getSrc(), dstDevice = htd.getDst();
  gpu::DeallocOp srcDealloc;
  for (auto user : srcDevice.getUsers()) {
    if (isa<gpu::DeallocOp>(user)) {
      srcDealloc = cast<gpu::DeallocOp>(user);
      break;
    }
  }
  gpu::AllocOp dstAlloc = cast<gpu::AllocOp>(dstDevice.getDefiningOp());
  dstDevice.replaceAllUsesWith(srcDevice);
  htd.erase();
  dstAlloc.erase();
  srcDealloc.erase();
}

static void FoldGPUMemcpy(gpu::MemcpyOp dth, gpu::MemcpyOp htd) {
  auto srcDevice = dth.getSrc(), dstDevice = htd.getDst();
  gpu::DeallocOp srcDealloc;
  for (auto user : srcDevice.getUsers()) {
    if (isa<gpu::DeallocOp>(user)) {
      srcDealloc = cast<gpu::DeallocOp>(user);
      break;
    }
  }
  gpu::AllocOp dstAlloc = cast<gpu::AllocOp>(dstDevice.getDefiningOp());

  dth.erase();
  srcDealloc.erase();
  dstDevice.replaceAllUsesWith(srcDevice);
  htd.erase();
  dstAlloc.erase();
}

void FoldGPUMemcpyPass::runOnOperation() {
  auto moduleOp = getOperation();
  bool change = false;
  llvm::DenseSet<gpu::MemcpyOp> waitToErase;
  do {
    change = false;
    llvm::DenseMap<Value, gpu::MemcpyOp> foldMap;
    std::pair<gpu::MemcpyOp, gpu::MemcpyOp> candidate;
    moduleOp.walk<WalkOrder::PreOrder>(
        [&foldMap, &candidate, &change](gpu::MemcpyOp memcpy) {
          if (memcpy.getSrc().getType().getMemorySpaceAsInt() == 1) {
            // device to host
            foldMap.insert({memcpy.getDst(), memcpy});
          } else {
            // host to device
            if (foldMap.count(memcpy.getSrc())) {
              candidate = {foldMap[memcpy.getSrc()], memcpy};
              change = true;
              return WalkResult::interrupt();
            }
          }
          return WalkResult::advance();
        });
    if (change) {
      waitToErase.insert(candidate.first);
      FoldGPUMemcpyhtd(candidate.first, candidate.second);
    }
  } while (change);
  for (auto cpy : waitToErase) {
    FoldGPUMemcpydth(cpy);
  }
}