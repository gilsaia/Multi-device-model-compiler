#include "multi-device-model-compiler/Conversion/ConvertMemrefToGPU/ConvertMemrefToGPU.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

namespace multi_device {
namespace conversion {

class DmaStartToGPU : public OpConversionPattern<memref::DmaStartOp> {
public:
  using OpConversionPattern<memref::DmaStartOp>::OpConversionPattern;
  using OpAdaptor = typename memref::DmaStartOp::Adaptor;
  LogicalResult
  matchAndRewrite(memref::DmaStartOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto memcpyOp =
        rewriter.create<gpu::MemcpyOp>(op.getLoc(), TypeRange(), ValueRange(),
                                       op.getDstMemRef(), op.getSrcMemRef());
    rewriter.replaceOp(op, memcpyOp);
    return success();
  }
};

class DmaWaitToGPU : public OpConversionPattern<memref::DmaWaitOp> {
public:
  using OpConversionPattern<memref::DmaWaitOp>::OpConversionPattern;
  using OpAdaptor = typename memref::DmaWaitOp::Adaptor;
  LogicalResult
  matchAndRewrite(memref::DmaWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

void populateMemrefDmaToGPUConversionPattern(ConversionTarget &target,
                                             RewritePatternSet &patterns,
                                             TypeConverter &TypeConverter,
                                             MLIRContext &ctx) {
  patterns.insert<DmaStartToGPU, DmaWaitToGPU>(TypeConverter, &ctx);
}
} // namespace conversion
} // namespace multi_device