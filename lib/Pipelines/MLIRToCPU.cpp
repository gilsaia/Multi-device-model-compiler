#include "multi-device-model-compiler/Pipelines/ConvertPipelines.h"

#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

void multi_device::pipelines::createMLIRToCPUPipeline(mlir::OpPassManager &pm) {
  pm.addPass(mlir::tosa::createTosaInferShapesPass());
  pm.addPass(mlir::tosa::createTosaLayerwiseConstantFoldPass());
  pm.addPass(mlir::tosa::createTosaMakeBroadcastablePass());
  pm.addPass(mlir::tosa::createTosaInferShapesPass());
  pm.addPass(mlir::tosa::createTosaValidationPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalgNamed());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalg());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToArith());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::arith::createArithExpandOpsPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToTensor());
  pm.addPass(mlir::createConvertTensorToLinalgPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
  pm.addPass(mlir::createLinalgBufferizePass());
  pm.addPass(mlir::tensor::createTensorBufferizePass());
  // mlir::bufferization::OneShotBufferizationOptions bufferOption;
  // bufferOption.allowReturnAllocs = true;
  // bufferOption.allowUnknownOps = true;
  // pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufferOption));
  pm.addPass(mlir::func::createFuncBufferizePass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
}