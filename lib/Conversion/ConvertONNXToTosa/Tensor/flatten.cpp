#include "multi-device-model-compiler/Conversion/ConvertONNXToTosa/ConvertONNXToTosa.h"

#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
class ONNXFlattenOpLoweringToTOSA : public OpConversionPattern<ONNXFlattenOp> {
public:
  using OpConversionPattern<ONNXFlattenOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ONNXFlattenOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputType = op.getResult().getType().cast<ShapedType>();
    auto reshape = rewriter.create<tosa::ReshapeOp>(
        op.getLoc(), outputType, op.getInput(), outputType.getShape());
    rewriter.replaceOp(op, reshape);
    return success();
  }
};
} // namespace

void multi_device::conversion::populateLoweringONNXFlattenOpToTOSAPattern(
    mlir::ConversionTarget &target, mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx) {
  patterns.insert<ONNXFlattenOpLoweringToTOSA>(typeConverter, ctx);
}