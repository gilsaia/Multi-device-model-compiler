#ifndef MULTI_DEVICE_MODEL_COMPILER_DIALECT_DEVICE_OPS_
#define MULTI_DEVICE_MODEL_COMPILER_DIALECT_DEVICE_OPS_

include "multi-device-model-compiler/Dialect/Device/IR/DeviceBase.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/Interfaces.td"
include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td"

def Device_GPUOffloadingAttr : Device_Attr<"GPUOffloading","gpu_offloading",[
    OffloadingTranslationAttrTrait,
    DeclareAttrInterfaceMethods<OffloadingLLVMTranslationAttrInterface>,
    ]> {
    let description=[{
        Device GPU Offloading handler,will output the ptx code,generate right kernel call.
    }];
    let parameters = (ins
        StringRefParameter<"Kernel name">:$kernel,
        OptionalParameter<"Attribute", "Target to select for embedding.">:$target
    );
    let assemblyFormat=[{
        $kernel (`<` $target^ `>`)?
    }];
    let genVerifyDecl = 1;
}

#endif // MULTI_DEVICE_MODEL_COMPILER_DIALECT_DEVICE_OPS_