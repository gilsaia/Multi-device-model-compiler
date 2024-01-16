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

#endif // MULTI_DEVICE_MODEL_COMPILER_CONVERSION_PASSES_