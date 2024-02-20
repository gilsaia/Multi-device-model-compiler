#include "multi-device-model-compiler/Conversion/ConvertTosaToTPU/ConvertTosaToTPU.h"

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Traits/Traits.h"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

namespace {
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
} // namespace

namespace {
class TosaClampLoweringToTPU : public OpConversionPattern<tosa::ClampOp> {
public:
  using OpConversionPattern<tosa::ClampOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::ClampOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto defineOp = input.getDefiningOp();
    if (mlir::isa<tosa::TosaOp>(defineOp)) {
      return rewriter.notifyMatchFailure(defineOp,
                                         "Wrong op ,can't be tosa op");
    }
    bool isRelu =
        op.getMinFp() == APFloat::getZero(op.getMinFp().getSemantics());
    if (isRelu && defineOp->hasTrait<tpu_mlir::trait::SupportFuseRelu>()) {
      // fuse relu
      double maxFp = op.getMaxFp().convertToDouble();
      if (op.getMaxFp() == APFloat::getLargest(op.getMaxFp().getSemantics())) {
        maxFp = -1;
      }
      defineOp->setAttr("do_relu", rewriter.getBoolAttr(true));
      defineOp->setAttr("relu_limit", rewriter.getF64FloatAttr(maxFp));
      rewriter.replaceOp(op, defineOp);
      return success();
    } else {
      double minFp = op.getMinFp().convertToDouble(),
             maxFp = op.getMaxFp().convertToDouble();
      auto clip = rewriter.create<tpu::ClipOp>(
          op.getLoc(), op.getOutput().getType(), op.getInput(),
          rewriter.getF64FloatAttr(minFp), rewriter.getF64FloatAttr(maxFp),
          nullptr);
      rewriter.replaceOp(op, clip);
      return success();
    }
  }
};
} // namespace

namespace multi_device {
namespace conversion {

void populateTosaElementWiseToTPUConversionPattern(ConversionTarget &target,
                                                   RewritePatternSet &patterns,
                                                   TypeConverter &TypeConverter,
                                                   MLIRContext &ctx) {
  patterns.insert<TosaAddLoweringToTPU>(TypeConverter, &ctx);
}
void populateTosaFuseElementWiseToTPUConversionPattern(
    ConversionTarget &target, RewritePatternSet &patterns,
    TypeConverter &TypeConverter, MLIRContext &ctx) {
  patterns.insert<TosaClampLoweringToTPU>(TypeConverter, &ctx, 2);
}
} // namespace conversion
} // namespace multi_device