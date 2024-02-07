#ifndef MULTI_DEVICE_MODEL_COMPILER_CONVERSION_CONVERTONNXTOTOSA_CONVERTONNXTOTOSA_H_
#define MULTI_DEVICE_MODEL_COMPILER_CONVERSION_CONVERTONNXTOTOSA_CONVERTONNXTOTOSA_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace multi_device {
#define GEN_PASS_DECL_FRONTENDTOTOSALOWERINGFIX
#include "multi-device-model-compiler/Conversion/Passes.h.inc"

namespace conversion {
void populateLoweringONNXReturnOpToTOSAPattern(
    mlir::ConversionTarget &target, mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx);
void populateLoweringONNXMatmulOpToTOSAPattern(
    mlir::ConversionTarget &target, mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx);
} // namespace conversion
} // namespace multi_device

#endif