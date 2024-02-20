#include "multi-device-model-compiler/Conversion/ConvertTosaToDevice/ConvertTosaToDevice.h"
#include "multi-device-model-compiler/Dialect/Device/IR/Device.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
class TosaConv2dLoweringToDevice : public OpConversionPattern<tosa::Conv2DOp> {
public:
  using OpConversionPattern<tosa::Conv2DOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::Conv2DOp op, OpAdaptor adaptor,
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

    auto weight = op.getWeight();
    auto weightType = weight.getType();
    llvm::SmallVector<int64_t, 4> weightDims{
        weightType.getDimSize(0), weightType.getDimSize(3),
        weightType.getDimSize(1), weightType.getDimSize(2)};
    auto transposeWeightType =
        RankedTensorType::get(weightDims, weightType.getElementType());
    auto permVal = rewriter.getI64TensorAttr({0, 3, 1, 2});
    auto perm =
        rewriter.create<tosa::ConstOp>(op.getLoc(), permVal.getType(), permVal);
    auto transposeWeight = rewriter.create<tosa::TransposeOp>(
        op.getLoc(), transposeWeightType, op.getWeight(), perm.getResult());
    auto originWeight = transposeWeight.getResult();

    auto outputTensor = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), originOutput.getType().getShape(),
        originOutput.getType().getElementType());
    auto conv2d = rewriter.create<multi_device::device::Conv2DOp>(
        op.getLoc(), TypeRange(), originInput, originWeight, op.getBias(),
        outputTensor.getResult(), Value(), ValueRange(),
        rewriter.getDenseI64ArrayAttr({transposeWeightType.getDimSize(2),
                                       transposeWeightType.getDimSize(3)}),
        op.getPadAttr(), op.getStrideAttr(), op.getDilationAttr(),
        rewriter.getBoolAttr(false));

    bool fuse = false;
    Operation *postOp;
    if ((transposeWeightType.getDimSize(2) == 1 &&
         transposeWeightType.getDimSize(3) == 1) ||
        !originOutput.hasOneUse()) {
      // if output not one use or kernel is o*i*1*1, don't fuse kernel.
    } else {
      Operation *toFuseOp;
      for (auto user : originOutput.getUsers()) {
        toFuseOp = user;
      }
      if (mlir::isa<tosa::AddOp>(toFuseOp)) {
        auto postadd = mlir::cast<tosa::AddOp>(toFuseOp);
        Value postaddVal;
        if (postadd.getInput1() == originOutput) {
          postaddVal = postadd.getInput2();
        } else {
          postaddVal = postadd.getInput1();
        }
        conv2d.getPostaddMutable().append(postaddVal);
        auto insertionPoint = rewriter.saveInsertionPoint();
        rewriter.setInsertionPoint(postadd);
        auto moveConv2d = rewriter.clone(*conv2d);
        rewriter.eraseOp(conv2d);
        conv2d = mlir::cast<multi_device::device::Conv2DOp>(moveConv2d);
        rewriter.restoreInsertionPoint(insertionPoint);
        fuse = true;
        postOp = postadd;
      } else if (mlir::isa<tosa::ClampOp>(toFuseOp)) {
        auto postClamp = mlir::cast<tosa::ClampOp>(toFuseOp);
        auto minApf = postClamp.getMinFp(), maxApf = postClamp.getMaxFp();
        if (maxApf == APFloat::getLargest(maxApf.getSemantics()) &&
            minApf == APFloat::getZero(minApf.getSemantics())) {
          conv2d.setContainRelu(true);
          fuse = true;
          postOp = postClamp;
        }
      }
    }
    if (fuse) {
      rewriter.eraseOp(transposeInput);
      rewriter.eraseOp(op);
      rewriter.eraseOp(transposeOutput);
      rewriter.replaceOp(postOp, conv2d.getOutput());
    } else {
      rewriter.eraseOp(transposeInput);
      rewriter.eraseOp(op);
      rewriter.replaceOp(transposeOutput, conv2d.getOutput());
    }
    return success();
  }
};
} // namespace

void multi_device::conversion::populateTosaConv2dLoweringConversionPattern(
    ConversionTarget &target, RewritePatternSet &patterns, MLIRContext &ctx) {
  target.addIllegalOp<tosa::Conv2DOp>();
  patterns.insert<TosaConv2dLoweringToDevice>(&ctx, 50);
}