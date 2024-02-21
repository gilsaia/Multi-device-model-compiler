#include "multi-device-model-compiler/Conversion/ConvertONNXToDevice/ConvertONNXToDevice.h"
#include "multi-device-model-compiler/Dialect/Device/IR/Device.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace {
class DetectMultiHeadAttentionLayer
    : public OpConversionPattern<ONNXSoftmaxOp> {
public:
  using OpConversionPattern<ONNXSoftmaxOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ONNXSoftmaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {}
};
} // namespace

void multi_device::conversion::populateDetectMultiHeadAttentionLayerPattern(
    ConversionTarget &target, RewritePatternSet &patterns,
    TypeConverter &TypeConverter, MLIRContext &ctx) {
  // for each softmaxop ,we try to fold it once
  llvm::DenseSet<Operation *> detected;
  target.addDynamicallyLegalOp<ONNXSoftmaxOp>([&detected](Operation *op) {
    if (detected.count(op))
      return true;
    detected.insert(op);
    return false;
  });
  patterns.insert<DetectMultiHeadAttentionLayer>(&ctx, 100);
}