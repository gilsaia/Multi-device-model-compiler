#include "multi-device-model-compiler/Conversion/ConvertONNXToTosa/ConvertONNXToTosa.h"

#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
class ONNXReduceMeanOpLoweringToTOSA
    : public OpConversionPattern<ONNXReduceMeanV13Op> {
public:
  using OpConversionPattern<ONNXReduceMeanV13Op>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ONNXReduceMeanV13Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputType = op.getData().getType().cast<RankedTensorType>();
    auto reduceAxes = op.getAxesAttr();
    if (inputType.getRank() != 4 || reduceAxes.size() != 2 ||
        op.getKeepdims() != 1) {
      return rewriter.notifyMatchFailure(op, "Wrong axes or keepdims");
    }
    for (auto axes : reduceAxes) {
      auto dim = axes.cast<IntegerAttr>().getInt();
      if (dim < inputType.getRank() - 2 || dim >= inputType.getRank()) {
        return rewriter.notifyMatchFailure(op, "Wrong dim");
      }
    }

    llvm::SmallVector<int64_t, 4> inputDims{
        inputType.getDimSize(0), inputType.getDimSize(2),
        inputType.getDimSize(3), inputType.getDimSize(1)};
    auto transposeType =
        RankedTensorType::get(inputDims, inputType.getElementType());
    auto permVal = rewriter.getI64TensorAttr({0, 2, 3, 1});
    auto perm =
        rewriter.create<tosa::ConstOp>(op.getLoc(), permVal.getType(), permVal);
    auto transpose = rewriter.create<tosa::TransposeOp>(
        op.getLoc(), transposeType, op.getData(), perm.getResult());

    llvm::SmallVector<int64_t, 4> outputDims{inputType.getDimSize(0), 1, 1,
                                             inputType.getDimSize(1)};
    auto avgOutType =
        RankedTensorType::get(outputDims, inputType.getElementType());
    auto avgpool2d = rewriter.create<tosa::AvgPool2dOp>(
        op.getLoc(), avgOutType, transpose.getResult(),
        rewriter.getDenseI64ArrayAttr(
            {inputType.getDimSize(2), inputType.getDimSize(3)}),
        rewriter.getDenseI64ArrayAttr({1, 1}),
        rewriter.getDenseI64ArrayAttr({0, 0, 0, 0}),
        TypeAttr::get(rewriter.getF32Type()));

    auto dePermVal = rewriter.getI64TensorAttr({0, 3, 1, 2});
    auto dePerm = rewriter.create<tosa::ConstOp>(
        op.getLoc(), dePermVal.getType(), dePermVal);
    auto deTranspose = rewriter.create<tosa::TransposeOp>(
        op.getLoc(), op.getResult().getType(), avgpool2d.getResult(),
        dePerm.getResult());
    rewriter.replaceOp(op, deTranspose);
    return success();
  }
};
} // namespace

void multi_device::conversion::populateLoweringONNXReduceMeanOpToTOSAPattern(
    mlir::ConversionTarget &target, mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx) {
  patterns.insert<ONNXReduceMeanOpLoweringToTOSA>(typeConverter, ctx);
}