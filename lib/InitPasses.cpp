#include "multi-device-model-compiler/InitPasses.h"

#include "mlir/Pass/Pass.h"

#include "Pass/Passes.hpp"

void multi_device::initONNXPasses() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::createScrubDisposablePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::createONNXOpTransformPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::createDecomposeONNXToONNXPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::createConvOptONNXToONNXPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::createONNXHybridTransformPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::createShapeInferencePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::createConstPropONNXToONNXPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::createInstrumentPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::createInstrumentONNXSignaturePass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::createSimplifyShapeRelatedOpsPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::createONNXDimAnalysisPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::createConvertONNXToTOSAPass();
  });
}