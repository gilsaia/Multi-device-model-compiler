#ifndef MULTI_DEVICE_MODEL_COMPILER_DIALECT_DEVICE_BASE_
#define MULTI_DEVICE_MODEL_COMPILER_DIALECT_DEVICE_BASE_

include "mlir/IR/DialectBase.td"
include "mlir/IR/EnumAttr.td"

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

#endif // MULTI_DEVICE_MODEL_COMPILER_DIALECT_DEVICE_BASE_