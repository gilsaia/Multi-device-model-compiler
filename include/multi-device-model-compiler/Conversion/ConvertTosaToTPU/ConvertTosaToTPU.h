#ifndef MULTI_DEVICE_MODEL_COMPILER_CONVERSION_CONVERTTOSATOTPU_CONVERTTOSATOTPU_H_
#define MULTI_DEVICE_MODEL_COMPILER_CONVERSION_CONVERTTOSATOTPU_CONVERTTOSATOTPU_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace multi_device {
std::unique_ptr<mlir::Pass> createConvertTosaToTPUPass();
namespace conversion {
void populateTosaElementWiseToTPUConversionPattern(
    mlir::ConversionTarget &target, mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &TypeConverter, mlir::MLIRContext &ctx);
}
} // namespace multi_device

#endif