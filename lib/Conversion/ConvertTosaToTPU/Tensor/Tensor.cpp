#include "multi-device-model-compiler/Conversion/ConvertTosaToTPU/ConvertTosaToTPU.h"

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

namespace multi_device {
namespace conversion {
class TosaConstLoweringToTPU : public OpConversionPattern<tosa::ConstOp> {
public:
  using OpConversionPattern<tosa::ConstOp>::OpConversionPattern;
  using OpAdaptor = typename tosa::ConstOp::Adaptor;

  LogicalResult
  matchAndRewrite(tosa::ConstOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tpu_mlir::top::WeightOp>(op, op.getType(),
                                                         ValueRange());
    return success();
  }
};

class TosaReshapeLoweringToTPU : public OpConversionPattern<tosa::ReshapeOp> {
public:
  using OpConversionPattern<tosa::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto reshape = rewriter.create<tpu::ReshapeOp>(
        op.getLoc(), op.getOutput().getType(), op.getInput1(), Value());
    rewriter.replaceOp(op, reshape);
    return success();
  }
};

void populateTosaTensorToTPUConversionPattern(ConversionTarget &target,
                                              RewritePatternSet &patterns,
                                              TypeConverter &TypeConverter,
                                              MLIRContext &ctx) {
  patterns.insert<TosaConstLoweringToTPU>(TypeConverter, &ctx);
  patterns.insert<TosaReshapeLoweringToTPU>(TypeConverter, &ctx);
}
} // namespace conversion
} // namespace multi_device