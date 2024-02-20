#include "multi-device-model-compiler/Conversion/ConvertTosaToLinalg/ConvertTosaToLinalgSaveTensor.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace multi_device {

#define GEN_PASS_DEF_TOSALOWERTOLINALGSAVETENSOR
#include "multi-device-model-compiler/Conversion/Passes.h.inc"

void populateTosaToLinalgConversionPattern(ConversionTarget &target,
                                           RewritePatternSet &patterns,

                                           MLIRContext &ctx) {
  conversion::populateTosaFullConnectToLinalgConversionPattern(target, patterns,
                                                               ctx);
  conversion::populateTosaClampToLinalgConversionPattern(target, patterns, ctx);
}
} // namespace multi_device

namespace {
class TosaToLinalgSaveTensorPass final
    : public multi_device::impl::TosaLowerToLinalgSaveTensorBase<
          TosaToLinalgSaveTensorPass> {
public:
  TosaToLinalgSaveTensorPass() = default;
  void runOnOperation() override final;
};
} // namespace

void TosaToLinalgSaveTensorPass::runOnOperation() {
  auto moduleOp = getOperation();

  RewritePatternSet patterns(&getContext());
  ConversionTarget target(getContext());
  target.addLegalDialect<linalg::LinalgDialect, tosa::TosaDialect,
                         tensor::TensorDialect, scf::SCFDialect>();

  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  multi_device::populateTosaToLinalgConversionPattern(target, patterns,
                                                      getContext());
  if (failed(applyFullConversion(moduleOp, target, std::move(patterns)))) {
    signalPassFailure();
  }
}