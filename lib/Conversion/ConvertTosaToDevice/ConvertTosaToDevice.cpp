#include "multi-device-model-compiler/Conversion/ConvertTosaToDevice/ConvertTosaToDevice.h"
#include "multi-device-model-compiler/Conversion/ConvertTosaToLinalg/ConvertTosaToLinalgSaveTensor.h"
#include "multi-device-model-compiler/Dialect/Device/IR/Device.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace multi_device {
#define GEN_PASS_DEF_TOSALOWERTODEVICE
#include "multi-device-model-compiler/Conversion/Passes.h.inc"

void populateTosaToDeviceConversionPattern(mlir::ConversionTarget &target,
                                           mlir::RewritePatternSet &patterns,
                                           mlir::MLIRContext &ctx,
                                           bool convertLinalg) {
  conversion::populateTosaFullConnectToMatmulConversionPattern(target, patterns,
                                                               ctx);
  conversion::populateTosaConv2dLoweringConversionPattern(target, patterns,
                                                          ctx);
  conversion::populateTosaPool2dLoweringConversionPattern(target, patterns,
                                                          ctx);
  if (convertLinalg) {
    conversion::populateTosaClampToLinalgConversionPattern(target, patterns,
                                                           ctx);
  }
}
} // namespace multi_device

namespace {
class TosaToDevicePass final
    : public multi_device::impl::TosaLowerToDeviceBase<TosaToDevicePass> {
public:
  using TosaLowerToDeviceBase::TosaLowerToDeviceBase;
  TosaToDevicePass(const multi_device::TosaLowerToDeviceOptions &options)
      : TosaLowerToDeviceBase(options) {}
  void runOnOperation() override final;
};
} // namespace

void TosaToDevicePass::runOnOperation() {
  auto moduleOp = getOperation();

  RewritePatternSet patterns(&getContext());
  ConversionTarget target(getContext());
  target.addLegalDialect<tosa::TosaDialect, multi_device::device::DeviceDialect,
                         arith::ArithDialect, tensor::TensorDialect,
                         linalg::LinalgDialect>();

  multi_device::populateTosaToDeviceConversionPattern(
      target, patterns, getContext(), useLinalgConvert);
  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
    signalPassFailure();
  }
}