#include "multi-device-model-compiler/Conversion/ConvertTosaToDevice/ConvertTosaToDevice.h"
#include "multi-device-model-compiler/Dialect/Device/IR/Device.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
class TosaMaxPool2dLoweringToDevice
    : public OpConversionPattern<tosa::MaxPool2dOp> {
public:
  using OpConversionPattern<tosa::MaxPool2dOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::MaxPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto transposeInput = mlir::cast<tosa::TransposeOp>(input.getDefiningOp());
    auto originInput = transposeInput.getInput1();
    auto output = op.getOutput();
    tosa::TransposeOp transposeOutput;
    for (auto user : output.getUsers()) {
      transposeOutput = mlir::cast<tosa::TransposeOp>(user);
    }
    auto originOutput = transposeOutput.getResult();

    auto outputTensor = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), originOutput.getType().getShape(),
        originOutput.getType().getElementType());
    auto pool = rewriter.create<multi_device::device::Pool2DOp>(
        op.getLoc(), TypeRange(), originInput, outputTensor.getResult(),
        ValueRange(), op.getKernelAttr(), op.getPadAttr(), op.getStrideAttr(),
        rewriter.getStringAttr("max"));

    rewriter.eraseOp(transposeInput);
    rewriter.eraseOp(op);
    rewriter.replaceOp(transposeOutput, pool.getOutput());
    return success();
  }
};

class TosaAvgPool2dLoweringToDevice
    : public OpConversionPattern<tosa::AvgPool2dOp> {
public:
  using OpConversionPattern<tosa::AvgPool2dOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::AvgPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto transposeInput = mlir::cast<tosa::TransposeOp>(input.getDefiningOp());
    auto originInput = transposeInput.getInput1();
    auto output = op.getOutput();
    tosa::TransposeOp transposeOutput;
    for (auto user : output.getUsers()) {
      transposeOutput = mlir::cast<tosa::TransposeOp>(user);
    }
    auto originOutput = transposeOutput.getResult();

    auto outputTensor = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), originOutput.getType().getShape(),
        originOutput.getType().getElementType());
    auto pool = rewriter.create<multi_device::device::Pool2DOp>(
        op.getLoc(), TypeRange(), originInput, outputTensor.getResult(),
        ValueRange(), op.getKernelAttr(), op.getPadAttr(), op.getStrideAttr(),
        rewriter.getStringAttr("avg"));

    rewriter.eraseOp(transposeInput);
    rewriter.eraseOp(op);
    rewriter.replaceOp(transposeOutput, pool.getOutput());
    return success();
  }
};
} // namespace

void multi_device::conversion::populateTosaPool2dLoweringConversionPattern(
    ConversionTarget &target, RewritePatternSet &patterns, MLIRContext &ctx) {
  target.addIllegalOp<tosa::MaxPool2dOp, tosa::AvgPool2dOp>();
  patterns.insert<TosaMaxPool2dLoweringToDevice, TosaAvgPool2dLoweringToDevice>(
      &ctx, 30);
}