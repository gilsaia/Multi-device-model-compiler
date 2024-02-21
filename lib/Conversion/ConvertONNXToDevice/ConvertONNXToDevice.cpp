#include "multi-device-model-compiler/Conversion/ConvertONNXToDevice/ConvertONNXToDevice.h"
#include "multi-device-model-compiler/Dialect/Device/IR/Device.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

namespace multi_device {
#define GEN_PASS_DEF_ONNXLOWERTODEVICE
#include "multi-device-model-compiler/Conversion/Passes.h.inc"
} // namespace multi_device

namespace {
class ONNXLowerToDevicePass final
    : public multi_device::impl::ONNXLowerToDeviceBase<ONNXLowerToDevicePass> {
public:
  ONNXLowerToDevicePass() = default;
  void runOnOperation() override final;
};
} // namespace

void ONNXLowerToDevicePass::runOnOperation() {
  auto moduleOp = getOperation();
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  ConversionTarget target(*context);

  TypeConverter typeConverter;

  target.addLegalDialect<multi_device::device::DeviceDialect,
                         tensor::TensorDialect, ONNXDialect>();

  multi_device::conversion::populateDetectMultiHeadAttentionLayerPattern(
      target, patterns, typeConverter, *context);

  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
    signalPassFailure();
  }
}
