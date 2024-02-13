#ifndef MULTI_DEVICE_MODEL_COMPILER_CONVERSION_CONVERTTOSATOTPU_CONVERTTOSATOTPU_H_
#define MULTI_DEVICE_MODEL_COMPILER_CONVERSION_CONVERTTOSATOTPU_CONVERTTOSATOTPU_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace multi_device {
#define GEN_PASS_DECL_TOSALOWERTOTPU
#include "multi-device-model-compiler/Conversion/Passes.h.inc"
namespace conversion {
void populateTosaMatmulToTPUConversionPattern(
    mlir::ConversionTarget &target, mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &TypeConverter, mlir::MLIRContext &ctx);
void populateTosaElementWiseToTPUConversionPattern(
    mlir::ConversionTarget &target, mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &TypeConverter, mlir::MLIRContext &ctx);
void populateTosaFuseElementWiseToTPUConversionPattern(
    mlir::ConversionTarget &target, mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &TypeConverter, mlir::MLIRContext &ctx);
void populateTosaTensorToTPUConversionPattern(
    mlir::ConversionTarget &target, mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &TypeConverter, mlir::MLIRContext &ctx);
} // namespace conversion
} // namespace multi_device

#endif