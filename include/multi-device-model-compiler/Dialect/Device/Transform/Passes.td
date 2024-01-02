#ifndef MULTI_DEVICE_MODEL_COMPILER_DIALECT_DEVICE_TRANSFORM_
#define MULTI_DEVICE_MODEL_COMPILER_DIALECT_DEVICE_TRANSFORM_

include "mlir/Pass/PassBase.td"

def AddDeviceTypeToFunc: Pass<"add-device-type-to-func","mlir::func::FuncOp">{
    let summary="Add Device type to func op.";
    let description=[{
        We need to know what device to optimize,so add device type is needed.
    }];
    let constructor="multi_device::device::createAddDeviceTypeToFuncPass()";
    let dependentDialects=["mlir::func::FuncDialect","multi_device::device::DeviceDialect"];

    let options=[
        Option<"deviceType","type","multi_device::device::DeviceType",
        "multi_device::device::DeviceType::unknown","Target device type to convert">
    ];
} 

#endif // MULTI_DEVICE_MODEL_COMPILER_DIALECT_DEVICE_TRANSFORM_