#include "multi-device-model-compiler/Dialect/Device/IR/Device.h"
#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

namespace multi_device {
namespace device {
#define GEN_PASS_DEF_DEVICEDATACOPYGENERATION
#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h.inc"
} // namespace device
} // namespace multi_device

namespace {
class DeviceDataCopyGenerationPass final
    : public multi_device::device::impl::DeviceDataCopyGenerationBase<
          DeviceDataCopyGenerationPass> {
public:
  using DeviceDataCopyGenerationBase::DeviceDataCopyGenerationBase;
  void runOnOperation() override final;
};
} // namespace

static gpu::AllocOp GenDataAllocAndCpy(Operation *op, OpBuilder &builder,
                                       Value oriVal, MemRefType memType,
                                       bool copy) {
  builder.setInsertionPoint(op);
  auto dMemType = MemRefType::get(memType.getShape(), memType.getElementType(),
                                  memType.getLayout().getAffineMap(), 1);
  auto alloc =
      builder.create<gpu::AllocOp>(op->getLoc(), dMemType, nullptr,
                                   ValueRange(), ValueRange(), ValueRange());
  if (copy) {
    builder.create<gpu::MemcpyOp>(op->getLoc(), TypeRange(), ValueRange(),
                                  alloc.getResult(0), oriVal);
  }
  builder.setInsertionPointAfter(op);
  builder.create<gpu::DeallocOp>(op->getLoc(), TypeRange(), ValueRange(),
                                 alloc.getResult(0));
  return alloc;
}

void DeviceDataCopyGenerationPass::runOnOperation() {
  auto moduleOp = getOperation();
  auto ctx = moduleOp.getContext();
  OpBuilder builder(ctx);
  moduleOp.walk([&ctx, &builder](multi_device::device::MatmulOp matmul) {
    builder.setInsertionPoint(matmul);
    auto inputType = matmul.getInput().getType().cast<MemRefType>(),
         weightType = matmul.getWeight().getType().cast<MemRefType>(),
         biasType = matmul.getBias().getType().cast<MemRefType>(),
         outputType = matmul.getOutput().getType().cast<MemRefType>();
    auto inputAlloc =
        GenDataAllocAndCpy(matmul, builder, matmul.getInput(), inputType, true);
    auto weightAlloc = GenDataAllocAndCpy(matmul, builder, matmul.getWeight(),
                                          weightType, true);
    auto biasAlloc =
        GenDataAllocAndCpy(matmul, builder, matmul.getBias(), biasType, true);
    auto outputAlloc = GenDataAllocAndCpy(matmul, builder, matmul.getOutput(),
                                          outputType, false);
    builder.setInsertionPointAfter(matmul);
    builder.create<gpu::MemcpyOp>(matmul.getLoc(), TypeRange(), ValueRange(),
                                  matmul.getOutput(), outputAlloc.getResult(0));
    matmul.setOperand(0, inputAlloc.getResult(0));
    matmul.setOperand(1, weightAlloc.getResult(0));
    matmul.setOperand(2, biasAlloc.getResult(0));
    matmul.setOperand(3, outputAlloc.getResult(0));
  });
}