#include "multi-device-model-compiler/Conversion/ConvertONNXToTosa/ConvertONNXToTosa.h"
#include "multi-device-model-compiler/Pipelines/ConvertPipelines.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "src/Pass/Passes.hpp"

using namespace mlir;

void multi_device::pipelines::createONNXToMLIRPipeline(
    mlir::OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createDecomposeONNXToONNXPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
  pm.addPass(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createConstPropONNXToONNXPass());
  // Simplify shape-related ops.
  pm.addPass(onnx_mlir::createSimplifyShapeRelatedOpsPass());
  // Clean dead code.
  pm.addPass(mlir::createSymbolDCEPass());
  // Replace every DisposableElementsAttr with DenseElementsAttr.
  pm.addPass(onnx_mlir::createScrubDisposablePass());
  pm.addPass(multi_device::createFrontendToTosaLoweringFix());
  // pm.addPass(mlir::createCSEPass());
}