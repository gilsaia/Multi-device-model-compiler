#include "multi-device-model-compiler/Conversion/ConvertONNXToTosa/ConvertONNXToTosa.h"
#include "multi-device-model-compiler/Conversion/Passes.h"
#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h"
#include "multi-device-model-compiler/Dialect/ONNX/Transform/Passes.h"
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
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

void multi_device::pipelines::createMLIRToCPUPipeline(mlir::OpPassManager &pm) {
  pm.addPass(multi_device::createONNXLowerToDevice());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(multi_device::createFrontendToTosaLoweringFix());
  device::AddDeviceTypeToFuncOptions deviceOptions;
  deviceOptions.deviceType = device::DeviceType::CPU;
  pm.addPass(
      multi_device::device::createAddDeviceTypeToFuncPass(deviceOptions));
  pm.addPass(mlir::tosa::createTosaInferShapesPass());
  // pm.addPass(mlir::tosa::createTosaLayerwiseConstantFoldPass());
  pm.addPass(mlir::tosa::createTosaMakeBroadcastablePass());
  pm.addPass(mlir::tosa::createTosaInferShapesPass());
  pm.addPass(mlir::tosa::createTosaValidationPass());

  pm.addPass(multi_device::createTosaLowerToDevice());
  pm.addPass(multi_device::createTosaLowerToLinalgSaveTensor());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalgNamed());
  pm.addPass(mlir::tosa::createTosaLayerwiseConstantFoldPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalg());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToArith());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::arith::createArithExpandOpsPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToTensor());
  pm.addPass(mlir::createConvertTensorToLinalgPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createLinalgDetensorizePass());

  pm.addPass(mlir::createCanonicalizerPass());
  // pm.addPass(mlir::createCSEPass());

  pm.addPass(mlir::bufferization::createEmptyTensorEliminationPass());
  pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
  pm.addPass(mlir::tensor::createTensorBufferizePass());
  pm.addPass(mlir::arith::createArithBufferizePass());
  pm.addPass(mlir::func::createFuncBufferizePass());
  pm.addPass(multi_device::device::createBufferizeOpWithAnalysis());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::bufferization::createBufferDeallocationPass());

  pm.addPass(mlir::bufferization::createBufferDeallocationSimplificationPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::bufferization::createBufferHoistingPass());
  pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
  pm.addPass(mlir::memref::createNormalizeMemRefsPass());
  pm.addPass(mlir::bufferization::createFinalizingBufferizePass());

  pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
  pm.addPass(mlir::affine::createAffineLoopNormalizePass());
  pm.addPass(mlir::affine::createAffineLoopInvariantCodeMotionPass());
  pm.addPass(mlir::affine::createAffineExpandIndexOpsPass());
  pm.addPass(mlir::affine::createSimplifyAffineStructuresPass());
  pm.addPass(mlir::affine::createLoopFusionPass(0, 0, true));
  pm.addPass(multi_device::device::createVectorizeAffineForDevice());
  pm.addPass(mlir::affine::createLoopCoalescingPass());
  pm.addPass(mlir::affine::createAffineLoopNormalizePass());
  pm.addPass(mlir::affine::createAffineExpandIndexOpsPass());
  pm.addPass(mlir::affine::createAffineScalarReplacementPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::affine::createSimplifyAffineStructuresPass());
  pm.addPass(mlir::affine::createLoopFusionPass(0, 0, true));

  pm.addPass(mlir::createLowerAffinePass());
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
  auto deviceToLLVMConfig = multi_device::ConvertDeviceToLLVMOptions();
  deviceToLLVMConfig.hostBarePtrCallConv = true;
  deviceToLLVMConfig.kernelBarePtrCallConv = true;
  pm.addPass(multi_device::createConvertDeviceToLLVM(deviceToLLVMConfig));
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(multi_device::createEliminateEntryPointPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}