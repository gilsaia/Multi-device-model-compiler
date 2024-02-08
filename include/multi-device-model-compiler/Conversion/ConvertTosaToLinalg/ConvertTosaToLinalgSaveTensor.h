#ifndef MULTI_DEVICE_MODEL_COMPILER_CONVERSION_CONVERTTOSATOLINALG_CONVERTTOSATOLINALGSAVETENSOR_H_
#define MULTI_DEVICE_MODEL_COMPILER_CONVERSION_CONVERTTOSATOLINALG_CONVERTTOSATOLINALGSAVETENSOR_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace multi_device {
#define GEN_PASS_DECL_TOSALOWERTOLINALGSAVETENSOR
#include "multi-device-model-compiler/Conversion/Passes.h.inc"
namespace conversion {
void populateTosaFullConnectToLinalgConversionPattern(
    mlir::ConversionTarget &target, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext &ctx);

void populateTosaClampToLinalgConversionPattern(
    mlir::ConversionTarget &target, mlir::RewritePatternSet &patterns,
    mlir::MLIRContext &ctx);
} // namespace conversion
} // namespace multi_device

#endif