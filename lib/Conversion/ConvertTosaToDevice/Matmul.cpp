#include "multi-device-model-compiler/Conversion/ConvertTosaToDevice/ConvertTosaToDevice.h"
#include "multi-device-model-compiler/Dialect/Device/IR/Device.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
class TosaFullConnectLoweringToDevice
    : public OpConversionPattern<tosa::FullyConnectedOp> {
public:
  using OpConversionPattern<tosa::FullyConnectedOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tosa::FullyConnectedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto input = op.getInput();
    auto inputShape = input.getType().cast<RankedTensorType>().getShape();
    if (mlir::isa<tosa::ReshapeOp>(input.getDefiningOp())) {
      auto reshape = mlir::cast<tosa::ReshapeOp>(input.getDefiningOp());
      auto reshapeInputType =
          reshape.getInput1().getType().cast<RankedTensorType>();
      if (reshapeInputType.getShape().back() ==
          op.getWeight().getType().getShape().back()) {
        input = reshape.getInput1();
        inputShape = input.getType().cast<RankedTensorType>().getShape();
        rewriter.eraseOp(reshape);
      }
    }

    auto output = op.getResult();
    auto outputResultType = output.getType().cast<RankedTensorType>();
    bool onlyReshape = false;
    tosa::ReshapeOp outputReshape;
    for (auto user : output.getUsers()) {
      outputReshape = mlir::dyn_cast<tosa::ReshapeOp>(user);
      if (!outputReshape) {
        onlyReshape = false;
        break;
      }
      if (outputReshape.getResult().getType().getRank() == inputShape.size()) {
        onlyReshape = true;
      }
    }
    if (onlyReshape) {
      output = outputReshape.getResult();
      outputResultType = output.getType().cast<RankedTensorType>();
    }

    auto outputTensor = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), outputResultType.getShape(),
        outputResultType.getElementType());

    auto weightType = op.getWeight().getType().cast<RankedTensorType>();
    llvm::SmallVector<int64_t, 2> weightDims{weightType.getShape()[1],
                                             weightType.getShape()[0]};
    auto transposeWeightType =
        RankedTensorType::get(weightDims, weightType.getElementType());
    auto permVal = rewriter.getI64TensorAttr({1, 0});
    auto perm =
        rewriter.create<tosa::ConstOp>(op.getLoc(), permVal.getType(), permVal);
    auto transpose = rewriter.create<tosa::TransposeOp>(
        op.getLoc(), transposeWeightType, op.getWeight(), perm.getResult());

    auto matmul = rewriter.create<multi_device::device::MatmulOp>(
        op.getLoc(), TypeRange(), input, transpose.getResult(), op.getBias(),
        outputTensor.getResult(), ValueRange());

    if (onlyReshape) {
      rewriter.eraseOp(op);
      rewriter.replaceOp(outputReshape, matmul.getOutput());
    } else {
      rewriter.replaceOp(op, matmul.getOutput());
    }
    return success();
  }
};
} // namespace

void multi_device::conversion::populateTosaFullConnectToMatmulConversionPattern(
    ConversionTarget &target, RewritePatternSet &patterns, MLIRContext &ctx) {
  target.addIllegalOp<tosa::FullyConnectedOp>();
  patterns.insert<TosaFullConnectLoweringToDevice>(&ctx, 50);
}