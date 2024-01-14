#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h"
#include "multi-device-model-compiler/Pipelines/ConvertPipelines.h"

#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

void multi_device::pipelines::createMLIRToGPUPipeline(mlir::OpPassManager &pm) {
  device::AddDeviceTypeToFuncOptions deviceOptions;
  deviceOptions.deviceType = device::DeviceType::GPU;
  pm.addPass(device::createAddDeviceTypeToFuncPass(deviceOptions));
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
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
  pm.addPass(mlir::createLinalgBufferizePass());
  pm.addPass(mlir::tensor::createTensorBufferizePass());
  pm.addPass(mlir::arith::createArithBufferizePass());
  pm.addPass(mlir::func::createFuncBufferizePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::bufferization::createBufferDeallocationPass());
  pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
  pm.addPass(mlir::createAffineExpandIndexOpsPass());
  pm.addPass(mlir::createAffineLoopNormalizePass());
  pm.addPass(mlir::createSimplifyAffineStructuresPass());
  auto affineVecConfig = mlir::AffineVectorizeOptions();
  std::vector<int64_t> vecSizes{32}, testFastestSizes{0};
  affineVecConfig.vectorSizes = vecSizes;
  affineVecConfig.fastestVaryingPattern = testFastestSizes;
  pm.addPass(mlir::createAffineVectorize(affineVecConfig));
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createAffineForToGPUPass());
  pm.addPass(mlir::createConvertVectorToGPUPass(true));
  pm.addPass(mlir::createGpuLauchSinkIndexComputationsPass());
  pm.addPass(mlir::createGpuKernelOutliningPass());
  pm.addPass(mlir::createGpuAsyncRegionPass());
}