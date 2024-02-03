#ifndef MULTI_DEVICE_MODEL_COMPILER_DIALECT_DEVICE_BASE_
#define MULTI_DEVICE_MODEL_COMPILER_DIALECT_DEVICE_BASE_

include "mlir/IR/DialectBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"

include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td"

def Device_Dialect: Dialect {
    let name="device";
    let cppNamespace="::multi_device::device";
    let summary=[{
        The device dialect used to describe device attr used for optimize model.
    }];
    let useDefaultAttributePrinterParser = 1;
}

def DeviceType : I64EnumAttr<
    "DeviceType","",
    [
        I64EnumAttrCase<"unknown",0,"unknown">,
        I64EnumAttrCase<"CPU",1,"cpu">,
        I64EnumAttrCase<"GPU",2,"gpu">,
        I64EnumAttrCase<"TPU",3,"tpu">,
    ]> {
    let cppNamespace="::multi_device::device";
} 

class Device_Attr<string attrName,string attrMnemonic,list<Trait> traits=[]>
    : AttrDef<Device_Dialect,attrName,traits>{
        let mnemonic=attrMnemonic;
}

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

#endif // MULTI_DEVICE_MODEL_COMPILER_DIALECT_DEVICE_BASE_