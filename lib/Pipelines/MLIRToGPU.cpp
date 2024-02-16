#include "multi-device-model-compiler/Conversion/ConvertDeviceToLLVM/ConvertDeviceToLLVM.h"
#include "multi-device-model-compiler/Conversion/ConvertGPUToNVVM/ConvertGPUToNVVM.h"
#include "multi-device-model-compiler/Conversion/ConvertMemrefToGPU/ConvertMemrefToGPU.h"
#include "multi-device-model-compiler/Conversion/ConvertTosaToLinalg/ConvertTosaToLinalgSaveTensor.h"
#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h"
#include "multi-device-model-compiler/Dialect/ONNX/Transform/Passes.h"
#include "multi-device-model-compiler/Pipelines/ConvertPipelines.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/OptimizeForNVVM.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/NVGPU/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

void multi_device::pipelines::createMLIRToGPUPipeline(mlir::OpPassManager &pm) {
  device::AddDeviceTypeToFuncOptions deviceOptions;
  deviceOptions.deviceType = device::DeviceType::GPU;
  pm.addPass(device::createAddDeviceTypeToFuncPass(deviceOptions));
  pm.addPass(multi_device::createEliminateEntryPointPass());
  pm.addPass(mlir::tosa::createTosaInferShapesPass());
  pm.addPass(mlir::tosa::createTosaLayerwiseConstantFoldPass());
  pm.addPass(mlir::tosa::createTosaMakeBroadcastablePass());
  pm.addPass(mlir::tosa::createTosaInferShapesPass());
  pm.addPass(mlir::tosa::createTosaValidationPass());

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
  pm.addPass(mlir::createCSEPass());

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
  pm.addPass(mlir::affine::createAffineDataCopyGenerationPass(0, 1, 0));
  pm.addPass(mlir::affine::createLoopFusionPass(1, 0, true));
  //   pm.addPass(multi_device::device::createVectorizeAffineForDevice());
  //   pm.addPass(mlir::affine::createLoopCoalescingPass());
  pm.addPass(multi_device::device::createCoalesceAffineForDevice());
  pm.addPass(mlir::affine::createSimplifyAffineStructuresPass());
  pm.addPass(mlir::affine::createAffineLoopNormalizePass());
  pm.addPass(mlir::affine::createAffineExpandIndexOpsPass());
  pm.addPass(mlir::affine::createAffineScalarReplacementPass());
  pm.addPass(mlir::affine::createLoopFusionPass(1, 0, true));

  pm.addPass(mlir::affine::createPipelineDataTransferPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::affine::createSimplifyAffineStructuresPass());
  pm.addPass(mlir::affine::createAffineParallelizePass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createParallelLoopSpecializationPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  pm.addPass(multi_device::device::createTilingScfParallelDevice());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createParallelLoopSpecializationPass());
  pm.addPass(mlir::createGpuMapParallelLoopsPass());
  pm.addPass(mlir::createParallelLoopToGpuPass());
  pm.addPass(multi_device::createConvertMemrefToGPU());
  pm.addPass(multi_device::device::createFoldGPUMemcpy());
  pm.addPass(multi_device::device::createEliminateGPUMemrefSpace());
  pm.addPass(mlir::createGpuLauchSinkIndexComputationsPass());
  pm.addPass(mlir::createGpuKernelOutliningPass());
  pm.addPass(multi_device::device::createCombineGPUKernel());
  pm.addPass(mlir::createGpuAsyncRegionPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      multi_device::device::createAsyncDependencyConvert());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  mlir::GpuNVVMAttachTargetOptions attachOptions;
  attachOptions.chip = "sm_70";
  attachOptions.features = "+ptx75";
  pm.addPass(mlir::createGpuNVVMAttachTarget(attachOptions));

  pm.addPass(mlir::createConvertVectorToGPUPass(true));
  pm.addPass(mlir::createConvertVectorToSCFPass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::LLVM::createRequestCWrappersPass());
  auto arithToLLVMConfig = mlir::ArithToLLVMConversionPassOptions();
  pm.addPass(mlir::createArithToLLVMConversionPass(arithToLLVMConfig));
  auto cfToLLVMConfig = mlir::ConvertControlFlowToLLVMPassOptions();
  pm.addPass(mlir::createConvertControlFlowToLLVMPass(cfToLLVMConfig));
  auto vecToLLVMConfig = mlir::ConvertVectorToLLVMPassOptions();
  pm.addPass(mlir::createConvertVectorToLLVMPass(vecToLLVMConfig));
  auto funcToLLVMConfig = mlir::ConvertFuncToLLVMPassOptions();
  pm.addPass(mlir::createConvertFuncToLLVMPass(funcToLLVMConfig));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // gpu module
  auto gpuToNVVMConfig = ConvertGpuOpsToNVVMOpsFixOptions();
  gpuToNVVMConfig.useBarePtrCallConv = true;
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(
      createConvertGpuOpsToNVVMOpsFix(gpuToNVVMConfig));
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::nvgpu::createOptimizeSharedMemoryPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::NVVM::createOptimizeForTargetPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createCSEPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::createReconcileUnrealizedCastsPass());

  auto memrefToLLVMConfig = mlir::FinalizeMemRefToLLVMConversionPassOptions();
  pm.addPass(
      mlir::createFinalizeMemRefToLLVMConversionPass(memrefToLLVMConfig));

  pm.addPass(mlir::createConvertNVVMToLLVMPass());
  auto deviceToLLVMConfig = multi_device::ConvertDeviceToLLVMOptions();
  deviceToLLVMConfig.hostBarePtrCallConv = true;
  deviceToLLVMConfig.kernelBarePtrCallConv = true;
  pm.addPass(multi_device::createConvertDeviceToLLVM(deviceToLLVMConfig));

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  auto offloadLLVMToGPUConfig =
      multi_device::device::OffloadingLLVMToGPUOptions();
  offloadLLVMToGPUConfig.compilationTarget = "llvm";
  pm.addPass(device::createOffloadingLLVMToGPU(offloadLLVMToGPUConfig));
}