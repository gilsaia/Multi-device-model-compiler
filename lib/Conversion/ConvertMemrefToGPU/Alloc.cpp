#include "multi-device-model-compiler/Conversion/ConvertMemrefToGPU/ConvertMemrefToGPU.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

namespace multi_device {
namespace conversion {

class MemrefAllocToGPU : public OpConversionPattern<memref::AllocOp> {
public:
  using OpConversionPattern<memref::AllocOp>::OpConversionPattern;
  using OpAdaptor = typename memref::AllocOp::Adaptor;
  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcMemref = op.getMemref();
    auto allocOp =
        rewriter.create<gpu::AllocOp>(op.getLoc(), srcMemref.getType(), nullptr,
                                      ValueRange(), ValueRange(), ValueRange());
    rewriter.replaceOp(op, allocOp);
    return success();
  }
};

class MemrefDeallocToGPU : public OpConversionPattern<memref::DeallocOp> {
public:
  using OpConversionPattern<memref::DeallocOp>::OpConversionPattern;
  using OpAdaptor = typename memref::DeallocOp::Adaptor;
  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<gpu::DeallocOp>(op, TypeRange(), ValueRange(),
                                                op.getMemref());
    return success();
  }
};

void populateMemrefAllocToGPUConversionPattern(ConversionTarget &target,
                                               RewritePatternSet &patterns,
                                               TypeConverter &TypeConverter,
                                               MLIRContext &ctx) {
  patterns.insert<MemrefAllocToGPU, MemrefDeallocToGPU>(TypeConverter, &ctx);
}
} // namespace conversion
} // namespace multi_device