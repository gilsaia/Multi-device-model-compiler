#include "multi-device-model-compiler/Conversion/ConvertDeviceToTPU/ConvertDeviceToTPU.h"
#include "multi-device-model-compiler/Dialect/Device/IR/Device.h"

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

namespace {
class DevicePool2dLoweringToTPU
    : public OpConversionPattern<multi_device::device::Pool2DOp> {
  using OpConversionPattern<
      multi_device::device::Pool2DOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(multi_device::device::Pool2DOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto output = adaptor.getOutput();
    auto outputType = output.getType();

    auto poolMode =
        (op.getMethod() == "max") ? tpu::PoolMode::Max : tpu::PoolMode::Avg;
    llvm::SmallVector<NamedAttribute> attrs;
    attrs.emplace_back(rewriter.getNamedAttr("count_include_pad",
                                             rewriter.getBoolAttr(false)));
    attrs.emplace_back(
        rewriter.getNamedAttr("do_relu", rewriter.getBoolAttr(false)));
    attrs.emplace_back(
        rewriter.getNamedAttr("keep_dims", rewriter.getBoolAttr(true)));
    attrs.emplace_back(rewriter.getNamedAttr(
        "kernel_shape", rewriter.getI64ArrayAttr(op.getKernel())));
    attrs.emplace_back(
        rewriter.getNamedAttr("pad_value", rewriter.getI64IntegerAttr(0)));
    attrs.emplace_back(
        rewriter.getNamedAttr("pads", rewriter.getI64ArrayAttr(op.getPad())));
    attrs.emplace_back(rewriter.getNamedAttr(
        "strides", rewriter.getI64ArrayAttr(op.getStride())));
    attrs.emplace_back(
        rewriter.getNamedAttr("relu_limit", rewriter.getF64FloatAttr(-1)));
    attrs.emplace_back(rewriter.getNamedAttr(
        "pool_mode", tpu::PoolModeAttr::get(getContext(), poolMode)));

    llvm::SmallVector<Value> args = {adaptor.getInput()};

    ValueRange operands(args);
    auto tpuPool2d = rewriter.create<tpu::Pool2DOp>(op.getLoc(), outputType,
                                                    operands, attrs);

    auto tensorOp = output.getDefiningOp();
    rewriter.replaceOp(tensorOp, tpuPool2d.getResult());
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void multi_device::conversion::populateDevicePool2dToTPUConversionPattern(
    ConversionTarget &target, RewritePatternSet &patterns,
    TypeConverter &TypeConverter, MLIRContext &ctx) {
  patterns.insert<DevicePool2dLoweringToTPU>(TypeConverter, &ctx, 20);
}