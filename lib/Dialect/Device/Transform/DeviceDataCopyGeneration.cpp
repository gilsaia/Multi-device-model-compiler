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
  moduleOp.walk([&ctx, &builder](multi_device::device::Conv2DOp conv2d) {
    builder.setInsertionPoint(conv2d);
    auto inputType = conv2d.getInput().getType().cast<MemRefType>(),
         weightType = conv2d.getWeight().getType().cast<MemRefType>(),
         biasType = conv2d.getBias().getType().cast<MemRefType>(),
         outputType = conv2d.getOutput().getType().cast<MemRefType>();
    auto inputAlloc =
        GenDataAllocAndCpy(conv2d, builder, conv2d.getInput(), inputType, true);
    auto weightAlloc = GenDataAllocAndCpy(conv2d, builder, conv2d.getWeight(),
                                          weightType, true);
    auto biasAlloc =
        GenDataAllocAndCpy(conv2d, builder, conv2d.getBias(), biasType, true);
    auto outputAlloc = GenDataAllocAndCpy(conv2d, builder, conv2d.getOutput(),
                                          outputType, false);
    if (conv2d.getPostadd()) {
      auto postAddAlloc = GenDataAllocAndCpy(
          conv2d, builder, conv2d.getPostadd(), outputType, true);
      conv2d.setOperand(4, postAddAlloc.getResult(0));
    }
    builder.setInsertionPointAfter(conv2d);
    builder.create<gpu::MemcpyOp>(conv2d.getLoc(), TypeRange(), ValueRange(),
                                  conv2d.getOutput(), outputAlloc.getResult(0));
    conv2d.setOperand(0, inputAlloc.getResult(0));
    conv2d.setOperand(1, weightAlloc.getResult(0));
    conv2d.setOperand(2, biasAlloc.getResult(0));
    conv2d.setOperand(3, outputAlloc.getResult(0));
  });
  moduleOp.walk([&ctx, &builder](multi_device::device::Pool2DOp pool2d) {
    builder.setInsertionPoint(pool2d);
    auto inputType = pool2d.getInput().getType().cast<MemRefType>(),
         outputType = pool2d.getOutput().getType().cast<MemRefType>();
    auto inputAlloc =
        GenDataAllocAndCpy(pool2d, builder, pool2d.getInput(), inputType, true);
    auto outputAlloc = GenDataAllocAndCpy(pool2d, builder, pool2d.getOutput(),
                                          outputType, false);
    builder.setInsertionPointAfter(pool2d);
    builder.create<gpu::MemcpyOp>(pool2d.getLoc(), TypeRange(), ValueRange(),
                                  pool2d.getOutput(), outputAlloc.getResult(0));
    pool2d.setOperand(0, inputAlloc.getResult(0));
    pool2d.setOperand(1, outputAlloc.getResult(0));
  });
}