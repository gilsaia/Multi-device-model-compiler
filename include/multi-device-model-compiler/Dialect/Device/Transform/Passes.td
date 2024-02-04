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

def EliminateGPUMemrefSpace:Pass<"eliminate-gpu-memref-space","mlir::ModuleOp"> {
    let summary="Eliminate Memref space in gpu alloc.";
    let description=[{
        In NVVM, There is a space identifier to identify global/shared/local memory,
        but now in gpu we need to eliminate it so it can translate correctly.
    }]; 
    let dependentDialects=["mlir::gpu::GPUDialect"];
}

def AsyncDependencyConvert:Pass<"async-dependency-convert","mlir::func::FuncOp"> {
    let summary="Convert gpu.wait to device.wait,with dependency change.";
    let description=[{
        gpu.wait will cause multi-stream creation,which we want to use event-based to 
        replace that.
    }];
    let dependentDialects=["multi_device::device::DeviceDialect"];
}

#endif // MULTI_DEVICE_MODEL_COMPILER_DIALECT_DEVICE_TRANSFORM_