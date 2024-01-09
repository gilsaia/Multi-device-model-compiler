#include "multi-device-model-compiler/Conversion/ConvertTosaToTPU/ConvertTosaToTPU.h"

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"

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

void populateTosaTensorToTPUConversionPattern(ConversionTarget &target,
                                              RewritePatternSet &patterns,
                                              TypeConverter &TypeConverter,
                                              MLIRContext &ctx) {
  patterns.insert<TosaConstLoweringToTPU>(TypeConverter, &ctx);
}
} // namespace conversion
} // namespace multi_device