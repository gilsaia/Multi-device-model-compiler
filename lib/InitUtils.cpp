#include "multi-device-model-compiler/InitUtils.h"
#include "multi-device-model-compiler/Conversion/ConvertTosaToTPU/ConvertTosaToTPU.h"
#include "multi-device-model-compiler/Conversion/Passes.h"
#include "multi-device-model-compiler/Pass/InitPasses.h"
#include "multi-device-model-compiler/Pipelines/ConvertPipelines.h"

#include "mlir/Pass/Pass.h"

#include "src/Pass/Passes.hpp"

void multi_device::initONNXPasses() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return onnx_mlir::createScrubDisposablePass();
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

void multi_device::initConvertPasses() {
  registerTosaLowerToTPU();
  registerConvertMemrefToGPU();
}

void multi_device::initConvertPassPipelines() {
  mlir::PassPipelineRegistration<>(
      "onnx-to-mlir", "Pipeline lowering ONNX-IR to MLIR",
      multi_device::pipelines::createONNXToMLIRPipeline);
  mlir::PassPipelineRegistration<>(
      "mlir-to-cpu", "Pipeline lowering TOSA to CPU code",
      multi_device::pipelines::createMLIRToCPUPipeline);
  mlir::PassPipelineRegistration<>(
      "mlir-to-tpu", "Pipeline lowering TOSA to TPU code",
      multi_device::pipelines::createMLIRToTPUPipeline);
  mlir::PassPipelineRegistration<>(
      "mlir-to-gpu", "Pipeline lowering TOSA to GPU code",
      multi_device::pipelines::createMLIRToGPUPipeline);
}

void multi_device::initMultiDevicePasses() {
  registerONNXPasses();
  device::registerDevicePasses();
}