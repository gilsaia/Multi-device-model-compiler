#ifndef MULTI_DEVICE_MODEL_COMPILER_CONVERSION_CONVERTDEVICETOTPU_CONVERTDEVICETOTPU_H_
#define MULTI_DEVICE_MODEL_COMPILER_CONVERSION_CONVERTDEVICETOTPU_CONVERTDEVICETOTPU_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace multi_device {
#define GEN_PASS_DECL_DEVICELOWERTOTPU
#include "multi-device-model-compiler/Conversion/Passes.h.inc"
namespace conversion {
void populateDeviceMatmulToTPUConversionPattern(
    mlir::ConversionTarget &target, mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &TypeConverter, mlir::MLIRContext &ctx);
void populateFuseClipOpToTPUConversionPattern(
    mlir::ConversionTarget &target, mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &TypeConverter, mlir::MLIRContext &ctx);
void populateDeviceConv2dToTPUConversionPattern(
    mlir::ConversionTarget &target, mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &TypeConverter, mlir::MLIRContext &ctx);
void populateDevicePool2dToTPUConversionPattern(
    mlir::ConversionTarget &target, mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &TypeConverter, mlir::MLIRContext &ctx);
} // namespace conversion
} // namespace multi_device

#endif