#include "multi-device-model-compiler/Conversion/ConvertDeviceToTPU/ConvertDeviceToTPU.h"
#include "multi-device-model-compiler/Dialect/Device/IR/Device.h"

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

namespace {
class DeviceConv2dLoweringToTPU
    : public OpConversionPattern<multi_device::device::Conv2DOp> {
public:
  using OpConversionPattern<
      multi_device::device::Conv2DOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(multi_device::device::Conv2DOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto output = adaptor.getOutput();
    auto outputType = output.getType();

    llvm::SmallVector<NamedAttribute> attrs;
    attrs.emplace_back(rewriter.getNamedAttr(
        "do_relu", rewriter.getBoolAttr(op.getContainRelu())));
    attrs.emplace_back(
        rewriter.getNamedAttr("coeff_merged", rewriter.getBoolAttr(false)));
    attrs.emplace_back(rewriter.getNamedAttr(
        "dilations", rewriter.getI64ArrayAttr(op.getDilation())));
    attrs.emplace_back(
        rewriter.getNamedAttr("group", rewriter.getI64IntegerAttr(1)));
    attrs.emplace_back(rewriter.getNamedAttr(
        "kernel_shape", rewriter.getI64ArrayAttr(op.getKernel())));
    attrs.emplace_back(
        rewriter.getNamedAttr("kernel_zp", rewriter.getI64IntegerAttr(0)));
    attrs.emplace_back(
        rewriter.getNamedAttr("pads", rewriter.getI64ArrayAttr(op.getPad())));
    attrs.emplace_back(rewriter.getNamedAttr(
        "strides", rewriter.getI64ArrayAttr(op.getStride())));
    attrs.emplace_back(
        rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(true)));
    attrs.emplace_back(
        rewriter.getNamedAttr("relu_limit", rewriter.getF64FloatAttr(-1)));

    llvm::SmallVector<Value> args = {adaptor.getInput(), adaptor.getWeight(),
                                     adaptor.getBias()};
    ValueRange operands(args);

    auto conv = rewriter.create<tpu::Conv2DOp>(op.getLoc(), outputType,
                                               operands, attrs);

    auto changeOutput = conv.getResult();

    if (adaptor.getPostadd()) {
      llvm::SmallVector<NamedAttribute> addAttrs;
      attrs.emplace_back(
          rewriter.getNamedAttr("do_relu", rewriter.getBoolAttr(false)));
      attrs.emplace_back(
          rewriter.getNamedAttr("relu_limit", rewriter.getF64FloatAttr(-1)));

      llvm::SmallVector<Value> addArgs = {changeOutput, adaptor.getPostadd()};
      ValueRange addOperands(addArgs);

      auto add = rewriter.create<tpu::AddOp>(op.getLoc(), outputType,
                                             addOperands, addAttrs);
      changeOutput = add.getResult();
    }

    auto tensorOp = output.getDefiningOp();
    rewriter.replaceOp(tensorOp, changeOutput);
    rewriter.eraseOp(op);

    return success();
  }
};
} // namespace

void multi_device::conversion::populateDeviceConv2dToTPUConversionPattern(
    ConversionTarget &target, RewritePatternSet &patterns,
    TypeConverter &TypeConverter, MLIRContext &ctx) {
  patterns.insert<DeviceConv2dLoweringToTPU>(&ctx, 20);
}