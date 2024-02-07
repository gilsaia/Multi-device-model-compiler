#include "multi-device-model-compiler/Conversion/ConvertONNXToTosa/ConvertONNXToTosa.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
class ONNXMatmulOpLoweringToTOSA : public OpConversionPattern<ONNXMatMulOp> {
public:
  using OpConversionPattern<ONNXMatMulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ONNXMatMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value biasVal;
    ONNXAddOp bias;
    for (auto user : op.getResult().getUsers()) {
      if (mlir::isa<ONNXAddOp>(user)) {
        bias = mlir::cast<ONNXAddOp>(user);
        if (bias.getA() == op.getResult()) {
          biasVal = bias.getB();
        } else {
          biasVal = bias.getA();
        }
        break;
      }
    }

    auto weightType = op.getB().getType().cast<RankedTensorType>();
    llvm::SmallVector<int64_t, 2> weightDims{weightType.getShape()[1],
                                             weightType.getShape()[0]};
    auto transposeWeightType =
        RankedTensorType::get(weightDims, weightType.getElementType());
    auto permVal = rewriter.getI64TensorAttr({1, 0});
    auto perm =
        rewriter.create<tosa::ConstOp>(op.getLoc(), permVal.getType(), permVal);
    auto transpose = rewriter.create<tosa::TransposeOp>(
        op.getLoc(), transposeWeightType, op.getB(), perm.getResult());

    auto inputVal = op.getA();
    auto inputType = op.getA().getType().cast<RankedTensorType>();
    auto inputShape = inputType.getShape();
    auto outputType = op.getResult().getType().cast<RankedTensorType>();
    if (inputShape.size() > 2) {
      llvm::SmallVector<int64_t, 2> dims{1, 1};
      for (size_t i = 0; i < inputShape.size() - 1; ++i) {
        dims[0] *= inputShape[i];
      }
      dims[1] *= inputShape.back();
      auto reshapeType =
          RankedTensorType::get(dims, inputType.getElementType());
      auto reshape = rewriter.create<tosa::ReshapeOp>(op.getLoc(), reshapeType,
                                                      inputVal, dims);
      inputVal = reshape.getResult();

      dims[1] = op.getB().getType().cast<RankedTensorType>().getShape().back();
      auto reOutputType =
          RankedTensorType::get(dims, outputType.getElementType());

      auto fullconnect = rewriter.create<tosa::FullyConnectedOp>(
          op.getLoc(), reOutputType, inputVal, transpose.getResult(), biasVal);

      auto afterReshape = rewriter.create<tosa::ReshapeOp>(
          op.getLoc(), outputType, fullconnect.getResult(),
          outputType.getShape());
      rewriter.eraseOp(op);
      rewriter.replaceOp(bias, afterReshape);
    } else {
      auto fullconnect = rewriter.create<tosa::FullyConnectedOp>(
          op.getLoc(), op.getResult().getType(), op.getA(),
          transpose.getResult(), biasVal);
      rewriter.eraseOp(op);
      rewriter.replaceOp(bias, fullconnect);
    }
    return success();
  }
};
} // namespace

void multi_device::conversion::populateLoweringONNXMatmulOpToTOSAPattern(
    mlir::ConversionTarget &target, mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx) {
  patterns.insert<ONNXMatmulOpLoweringToTOSA>(typeConverter, ctx);
}