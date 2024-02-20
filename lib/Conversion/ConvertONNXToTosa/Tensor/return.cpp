#include "multi-device-model-compiler/Conversion/ConvertONNXToTosa/ConvertONNXToTosa.h"

#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
class ONNXReturnOpLoweringToTOSA : public OpConversionPattern<ONNXReturnOp> {
public:
  using OpConversionPattern<ONNXReturnOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ONNXReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto re = rewriter.create<func::ReturnOp>(op.getLoc(), op.getOperands());
    rewriter.replaceOp(op, re);
    return success();
  }
};
} // namespace

void multi_device::conversion::populateLoweringONNXReturnOpToTOSAPattern(
    mlir::ConversionTarget &target, mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx) {
  patterns.insert<ONNXReturnOpLoweringToTOSA>(typeConverter, ctx);
}