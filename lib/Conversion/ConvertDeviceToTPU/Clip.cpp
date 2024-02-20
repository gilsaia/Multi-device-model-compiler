#include "multi-device-model-compiler/Conversion/ConvertDeviceToTPU/ConvertDeviceToTPU.h"
#include "multi-device-model-compiler/Conversion/ConvertTosaToTPU/ConvertTosaToTPU.h"

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

namespace {
class TPUClipFuse : public OpConversionPattern<tpu::ClipOp> {
public:
  using OpConversionPattern<tpu::ClipOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tpu::ClipOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult
TPUClipFuse::matchAndRewrite(tpu::ClipOp op, OpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter) const {
  auto defineOp = op.getInput().getDefiningOp();
  defineOp->setAttr("do_relu", rewriter.getBoolAttr(true));
  defineOp->setAttr("relu_limit", rewriter.getF64FloatAttr(-1));

  rewriter.replaceOp(op, {op.getInput()});
  return success();
}

void multi_device::conversion::populateFuseClipOpToTPUConversionPattern(
    mlir::ConversionTarget &target, mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &TypeConverter, mlir::MLIRContext &ctx) {
  target.addDynamicallyLegalOp<tpu::ClipOp>([](Operation *op) {
    auto clip = cast<tpu::ClipOp>(op);
    if (clip.getMin() == APFloat::getZero(clip.getMin().getSemantics()) &&
        clip.getInput()
            .getDefiningOp()
            ->hasTrait<tpu_mlir::trait::SupportFuseRelu>()) {
      return false;
    }
    return true;
  });
  patterns.insert<TPUClipFuse>(TypeConverter, &ctx, 10);
}