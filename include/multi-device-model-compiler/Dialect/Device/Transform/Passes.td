#ifndef MULTI_DEVICE_MODEL_COMPILER_DIALECT_DEVICE_TRANSFORM_
#define MULTI_DEVICE_MODEL_COMPILER_DIALECT_DEVICE_TRANSFORM_

include "mlir/Pass/PassBase.td"

def AddDeviceTypeToFunc: Pass<"add-device-type-to-func","mlir::ModuleOp">{
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

def OffloadingLLVMToGPU: Pass<"offload-gpu-module",""> {
    let summary="Translate gpu module to binary with self handler.";
    let description=[{
        GPU ModuleToBinary Pass need a handler to control convert process,to do this,
        we need to build a attr which need in pass to build dynamic.

        The `format` argument can have the following values:
        1. `offloading`, `llvm`: producing an offloading representation.
        2. `assembly`, `isa`: producing assembly code.
        3. `binary`, `bin`: producing binaries.
    }];
    let options = [
    Option<"toolkitPath", "toolkit", "std::string", [{""}],
           "Toolkit path.">,
    ListOption<"linkFiles", "l", "std::string",
           "Extra files to link to.">,
    Option<"cmdOptions", "opts", "std::string", [{""}],
           "Command line options to pass to the tools.">,
    Option<"kernelName","name","std::string",[{"ops"}],"Kernel name to generate.">,
    Option<"compilationTarget", "format", "std::string", [{"bin"}],
           "The target representation of the compilation process.">
    ];
}

#endif // MULTI_DEVICE_MODEL_COMPILER_DIALECT_DEVICE_TRANSFORM_