#include "multi-device-model-compiler/Conversion/ConvertDeviceToTPU/ConvertDeviceToTPU.h"
#include "multi-device-model-compiler/Dialect/Device/IR/Device.h"

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

namespace {
class DeviceMatmulLoweringToTPU
    : public OpConversionPattern<multi_device::device::MatmulOp> {
public:
  using OpConversionPattern<
      multi_device::device::MatmulOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(multi_device::device::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto output = adaptor.getOutput();
    auto outputType = output.getType();

    llvm::SmallVector<NamedAttribute> attrs;
    attrs.emplace_back(
        rewriter.getNamedAttr("do_relu", rewriter.getBoolAttr(false)));
    attrs.emplace_back(
        rewriter.getNamedAttr("keep_dims", rewriter.getBoolAttr(true)));
    attrs.emplace_back(
        rewriter.getNamedAttr("hdim_is_batch", rewriter.getBoolAttr(false)));
    attrs.emplace_back(
        rewriter.getNamedAttr("left_transpose", rewriter.getBoolAttr(false)));
    attrs.emplace_back(
        rewriter.getNamedAttr("output_transpose", rewriter.getBoolAttr(false)));
    attrs.emplace_back(
        rewriter.getNamedAttr("right_transpose", rewriter.getBoolAttr(false)));

    llvm::SmallVector<Value> args = {adaptor.getInput(), adaptor.getWeight(),
                                     adaptor.getBias()};
    ValueRange operands(args);
    auto tpuMatmul = rewriter.create<tpu::MatMulOp>(op.getLoc(), outputType,
                                                    operands, attrs);
    auto tensorOp = output.getDefiningOp();
    rewriter.replaceOp(tensorOp, tpuMatmul.getResult());
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void multi_device::conversion::populateDeviceMatmulToTPUConversionPattern(
    ConversionTarget &target, RewritePatternSet &patterns,
    TypeConverter &TypeConverter, MLIRContext &ctx) {
  patterns.insert<DeviceMatmulLoweringToTPU>(TypeConverter, &ctx, 20);
}