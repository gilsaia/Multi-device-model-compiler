#ifndef MULTI_DEVICE_MODEL_COMPILER_CONVERSION_CONVERTMEMREFTOGPU_H_
#define MULTI_DEVICE_MODEL_COMPILER_CONVERSION_CONVERTMEMREFTOGPU_H_

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace multi_device {
#define GEN_PASS_DECL_CONVERTMEMREFTOGPU
#include "multi-device-model-compiler/Conversion/Passes.h.inc"
namespace conversion {
void populateMemrefAllocToGPUConversionPattern(
    mlir::ConversionTarget &target, mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &TypeConverter, mlir::MLIRContext &ctx);
void populateMemrefDmaToGPUConversionPattern(mlir::ConversionTarget &target,
                                             mlir::RewritePatternSet &patterns,
                                             mlir::TypeConverter &TypeConverter,
                                             mlir::MLIRContext &ctx);
} // namespace conversion
} // namespace multi_device

#endif