#ifndef MULTI_DEVICE_MODEL_COMPILER_CONVERSION_CONVERTONNXTODEVICE_CONVERTONNXTODEVICE_H_
#define MULTI_DEVICE_MODEL_COMPILER_CONVERSION_CONVERTONNXTODEVICE_CONVERTONNXTODEVICE_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace multi_device {
#define GEN_PASS_DECL_ONNXLOWERTODEVICE
#include "multi-device-model-compiler/Conversion/Passes.h.inc"
namespace conversion {
void populateDetectMultiHeadAttentionLayerPattern(
    mlir::ConversionTarget &target, mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &TypeConverter, mlir::MLIRContext &ctx);
}
} // namespace multi_device

#endif