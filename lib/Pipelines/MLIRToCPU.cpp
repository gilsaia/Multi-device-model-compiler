#include "multi-device-model-compiler/Pipelines/ConvertPipelines.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
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
  // pm.addPass(mlir::createLinalgGeneralizationPass());
  // pm.addPass(mlir::createLinalgElementwiseOpFusionPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
  pm.addPass(mlir::createLinalgBufferizePass());
  pm.addPass(mlir::tensor::createTensorBufferizePass());
  pm.addPass(mlir::func::createFuncBufferizePass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  pm.addPass(mlir::bufferization::createBufferDeallocationPass());
  pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
  pm.addPass(mlir::createAffineLoopNormalizePass());
  // pm.addPass(mlir::createLoopCoalescingPass());
  // pm.addPass(mlir::createLoopTilingPass());
  // pm.addPass(mlir::createLoopUnrollPass());
  // pm.addPass(mlir::createAffineParallelizePass());
  pm.addPass(mlir::createSimplifyAffineStructuresPass());
  auto affineVecConfig = mlir::AffineVectorizeOptions();
  std::vector<int64_t> vecSizes{32}, testFastestSizes{0};
  affineVecConfig.vectorSizes = vecSizes;
  affineVecConfig.fastestVaryingPattern = testFastestSizes;
  pm.addPass(mlir::createAffineVectorize(affineVecConfig));
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::bufferization::createFinalizingBufferizePass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::LLVM::createRequestCWrappersPass());
  auto arithToLLVMConfig = mlir::ArithToLLVMConversionPassOptions();
  pm.addPass(mlir::createArithToLLVMConversionPass(arithToLLVMConfig));
  auto cfToLLVMConfig = mlir::ConvertControlFlowToLLVMPassOptions();
  pm.addPass(mlir::createConvertControlFlowToLLVMPass(cfToLLVMConfig));
  auto vecToLLVMConfig = mlir::ConvertVectorToLLVMPassOptions();
  vecToLLVMConfig.x86Vector = true;
  pm.addPass(mlir::createConvertVectorToLLVMPass(vecToLLVMConfig));
  auto memrefToLLVMConfig = mlir::FinalizeMemRefToLLVMConversionPassOptions();
  pm.addPass(
      mlir::createFinalizeMemRefToLLVMConversionPass(memrefToLLVMConfig));
  auto funcToLLVMConfig = mlir::ConvertFuncToLLVMPassOptions();
  pm.addPass(mlir::createConvertFuncToLLVMPass(funcToLLVMConfig));
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}