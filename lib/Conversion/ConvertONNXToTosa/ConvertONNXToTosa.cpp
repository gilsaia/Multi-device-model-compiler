#include "multi-device-model-compiler/Conversion/ConvertONNXToTosa/ConvertONNXToTosa.h"

#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"

#include <optional>

using namespace mlir;

namespace multi_device {
void populateONNXToTOSAConversionPattern(ConversionTarget &target,
                                         RewritePatternSet &patterns,
                                         TypeConverter &typeConverter,
                                         MLIRContext *ctx) {
  // Math
  onnx_mlir::populateLoweringONNXElementwiseOpToTOSAPattern(target, patterns,
                                                            typeConverter, ctx);
  onnx_mlir::populateLoweringONNXReduceMeanOpToTOSAPattern(target, patterns,
                                                           typeConverter, ctx);
  onnx_mlir::populateLoweringONNXGemmOpToTOSAPattern(target, patterns,
                                                     typeConverter, ctx);
  onnx_mlir::populateLoweringONNXSoftmaxOpToTOSAPattern(target, patterns,
                                                        typeConverter, ctx);
  onnx_mlir::populateLoweringONNXConvOpToTOSAPattern(target, patterns,
                                                     typeConverter, ctx);
  // NN
  onnx_mlir::populateLoweringONNXMaxPoolSingleOutOpToTOSAPattern(
      target, patterns, typeConverter, ctx);
  // Tensor
  onnx_mlir::populateLoweringONNXConstOpToTOSAPattern(target, patterns,
                                                      typeConverter, ctx);
  onnx_mlir::populateLoweringONNXReshapeOpToTOSAPattern(target, patterns,
                                                        typeConverter, ctx);
}

// Performs lowering to TOSA dialect
struct FrontendToTosaLoweringFixPass
    : public PassWrapper<FrontendToTosaLoweringFixPass,
                         OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "convert-onnx-to-tosa-fix"; }

  StringRef getDescription() const override {
    return "Lower frontend ops to TOSA dialect with dialect registry.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::tosa::TosaDialect>();
  }

  FrontendToTosaLoweringFixPass() = default;
  FrontendToTosaLoweringFixPass(const FrontendToTosaLoweringFixPass &pass)
      : PassWrapper<FrontendToTosaLoweringFixPass, OperationPass<ModuleOp>>() {}

  void runOnOperation() final;
};

void FrontendToTosaLoweringFixPass::runOnOperation() {
  ModuleOp module = getOperation();
  // Define final conversion target
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  ConversionTarget target(*context);

  // We use the type converter to legalize types before any conversion patterns
  // are executed. This ensures that we do not need to trigger separate
  // conversion failures. Quantized types are not supported right now.
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) -> std::optional<Type> {
    if (onnx_mlir::isTOSASignedInt(type) || onnx_mlir::isTOSAFloat(type) ||
        type.isa<NoneType>())
      return type;
    return std::nullopt;
  });
  typeConverter.addConversion([&](TensorType type) -> std::optional<Type> {
    if (typeConverter.isLegal(type.getElementType()))
      return type;
    return std::nullopt;
  });

  // Define legal dialects and operations
  target.addLegalDialect<mlir::tosa::TosaDialect, func::FuncDialect,
                         mlir::arith::ArithDialect>();

  // Define patterns
  populateONNXToTOSAConversionPattern(target, patterns, typeConverter, context);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}
} // namespace multi_device

std::unique_ptr<mlir::Pass> multi_device::createConvertONNXToTOSAFixPass() {
  return std::make_unique<FrontendToTosaLoweringFixPass>();
}