#include "multi-device-model-compiler/Conversion/ConvertTosaToTPU/ConvertTosaToTPU.h"

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

namespace multi_device {
namespace conversion {
class TosaAddLoweringToTPU : public OpConversionPattern<tosa::AddOp> {
public:
  using OpConversionPattern<tosa::AddOp>::OpConversionPattern;
  using OpAdaptor = typename tosa::AddOp::Adaptor;
  LogicalResult
  matchAndRewrite(tosa::AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type outputs = op.getType();

    std::vector<NamedAttribute> attrs;
    attrs.emplace_back(
        rewriter.getNamedAttr("do_relu", rewriter.getBoolAttr(false)));
    attrs.emplace_back(
        rewriter.getNamedAttr("relu_limit", rewriter.getF64FloatAttr(-1)));
    rewriter.replaceOpWithNewOp<tpu::AddOp>(op, outputs, adaptor.getOperands(),
                                            attrs);

    return success();
  }
};

void populateTosaElementWiseToTPUConversionPattern(ConversionTarget &target,
                                                   RewritePatternSet &patterns,
                                                   TypeConverter &TypeConverter,
                                                   MLIRContext &ctx) {
  patterns.insert<TosaAddLoweringToTPU>(TypeConverter, &ctx);
}
} // namespace conversion
} // namespace multi_device