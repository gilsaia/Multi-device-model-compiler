#ifndef MULTI_DEVICE_MODEL_COMPILER_CONVERSION_PASSES_
#define MULTI_DEVICE_MODEL_COMPILER_CONVERSION_PASSES_

include "mlir/Pass/PassBase.td"

def ConvertGpuOpsToNVVMOpsFix : Pass<"convert-gpu-to-nvvm-fix", "mlir::gpu::GPUModuleOp"> {
  let summary = "Generate NVVM operations for gpu operations with full options";
  let dependentDialects = [
    "mlir::cf::ControlFlowDialect",
    "mlir::memref::MemRefDialect",
    "mlir::NVVM::NVVMDialect",
  ];
  let options = [
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
    Option<"hasRedux", "has-redux", "bool", /*default=*/"false",
           "Target gpu supports redux">,
    Option<"useBarePtrCallConv", "use-bare-ptr-memref-call-conv", "bool",
           /*default=*/"false",
           "Replace memref arguments in GPU functions with bare pointers. "
           "All memrefs must have static shape.">,
    Option<"useOpaquePointers", "use-opaque-pointers", "bool",
                   /*default=*/"true", "Generate LLVM IR using opaque pointers "
                   "instead of typed pointers">,
  ];
}

def ConvertMemrefToGPU : Pass<"convert-memref-to-gpu","mlir::ModuleOp"> {
       let summary = "Generate GPU operations for memref copy and alloc/dealloc in gpu space";
       let dependentDialects = [
              "mlir::gpu::GPUDialect"
       ];
}

def ConvertDeviceToLLVM:Pass<"convert-device-to-llvm","mlir::ModuleOp"> {
       let options = [
       Option<"hostBarePtrCallConv", "use-bare-pointers-for-host", "bool",
              /*default=*/"false",
              "Use bare pointers to pass memref arguments to host functions. "
              "All memrefs must have static shape.">,
       Option<"kernelBarePtrCallConv", "use-bare-pointers-for-kernels", "bool",
              /*default=*/"false",
              "Use bare pointers to pass memref arguments to kernels. "
              "The kernel must use the same setting for this option."
              >,
       Option<"gpuBinaryAnnotation", "gpu-binary-annotation", "std::string",
                     /*default=*/"mlir::gpu::getDefaultGpuBinaryAnnotation()",
                     "Annotation attribute string for GPU binary"
                     >,
       Option<"useOpaquePointers", "use-opaque-pointers", "bool",
                     /*default=*/"true", "Generate LLVM IR using opaque pointers "
                     "instead of typed pointers">,
       ];
       let summary = "Generate llvm ops from gpu/device.";
       let dependentDialects = [
              "mlir::LLVM::LLVMDialect",
              "mlir::memref::MemRefDialect",
       ];
}

def TosaLowerToTPU: Pass<"convert-tosa-to-tpu","mlir::func::FuncOp">{
       let summary="Lower tosa ops to TOP/TPU dialect.";
       let dependentDialects = [
              "tpu_mlir::top::TopDialect",
              "tpu_mlir::tpu::TpuDialect"
       ];
}

def TosaLowerToLinalgSaveTensor: Pass<"convert-tosa-to-linalg-save-tensor","mlir::ModuleOp"> {
       let summary = "Lower tosa ops to Linalg,which try to reuse tensor.";
       let dependentDialects = [
              "mlir::linalg::LinalgDialect",
              "mlir::arith::ArithDialect",
              "mlir::math::MathDialect",
              "mlir::tensor::TensorDialect",
              "mlir::scf::SCFDialect"
       ];
}

def TosaLowerToDevice: Pass<"convert-tosa-to-device","mlir::ModuleOp"> {
       let summary = "Lower tosa op to device op,with specific op accerlate.";
       let dependentDialects = [
              "mlir::linalg::LinalgDialect",
              "multi_device::device::DeviceDialect",
              "mlir::tensor::TensorDialect",
              "mlir::tosa::TosaDialect",
       ];
       let options = [
       Option<"useLinalgConvert", "use-linalg-convert", "bool",
              /*default=*/"true", "convert op to linalg op">
       ];
}

def DeviceLowerToTPU: Pass<"convert-device-to-tpu","mlir::ModuleOp"> {
       let summary = "Lower device op to tpu op.";
       let dependentDialects = [
              "tpu_mlir::top::TopDialect",
              "tpu_mlir::tpu::TpuDialect"
       ];
}

def FrontendToTosaLoweringFix: Pass<"convert-onnx-to-tosa-fix","mlir::ModuleOp"> {
       let summary="Lower frontend ops to TOSA dialect with dialect registry.";
       let dependentDialects = [
              "mlir::tosa::TosaDialect"
       ];
}

#endif // MULTI_DEVICE_MODEL_COMPILER_CONVERSION_PASSES_