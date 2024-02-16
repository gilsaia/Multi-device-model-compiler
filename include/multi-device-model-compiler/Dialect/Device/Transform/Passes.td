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

def CombineGPUKernel:Pass<"combine-gpu-kernel","mlir::ModuleOp"> {
    let summary="Combine different gpu kernel in one module,so we can use only one file to save it.";
    let dependentDialects=["mlir::gpu::GPUDialect"];
}

def FoldGPUMemcpy:Pass<"fold-gpu-memcpy","mlir::ModuleOp"> {
    let summary="Fold useless memcpy like gpu-cpu-gpu without change.";
    let dependentDialects=["mlir::gpu::GPUDialect","mlir::memref::MemRefDialect"];
}

def DeviceDataCopyGeneration:Pass<"device-data-copy-generate","mlir::ModuleOp"> {
    let summary="Generate data copy op in gpu mode for device ops.";
    let dependentDialects=["mlir::gpu::GPUDialect","mlir::memref::MemRefDialect"];
}

def TilingScfParallelDevice:Pass<"tiling-scf-parallel-device","mlir::ModuleOp"> {
    let summary="Find a proper tiling size for scf-parallel loop,to best use device power.";
    let description=[{
        This pass will try to find a proper tiling size with the information of device/data.
        for cpu, we assume that device better for 16-elements inside block.
        for gpu, we assume that device better for m-block-1024-thread, thread num can decrease
        when data size is small,but better with a 2^n mode.
    }];
    let dependentDialects=["mlir::scf::SCFDialect"];
    let options = [
    Option<"totalTilingSize", "tiling-size", "int64_t", /*default=*/"1024",
           "total tiling size for different device.">,
    ];
}

def BufferizeOpWithAnalysis:Pass<"bufferize-op-with-analysis","mlir::ModuleOp"> {
    let summary = "Use one-shot bufferize methods to buffer op that will not generate useless copy.";
    let description=[{
        This pass will use one-shot bufferize methods to buffer op,
        which in origin linalg op will generate copy-on-use first that is useless.
    }];
}

def VectorizeAffineForDevice:Pass<"vectorize-affine-for-device","mlir::ModuleOp"> {
    let summary = "Vectorize affine for in different ways,try to find better vector size.";
    let description=[{
        This pass will vectorize affine.for op, which will try to utilize inner-most for in cpu mode,
        will try to utilize tensor core for async device(like gpu)
    }];
    let options= [
        Option<"tensorCoreMode","tensor-core-mode","bool","false","vectorize in tensor core mode">,
    ];
}

def CoalesceAffineForDevice:Pass<"coalesce-affine-for-device","mlir::ModuleOp"> {
    let summary = "coalesce affine for with check load dim.";
}

#endif // MULTI_DEVICE_MODEL_COMPILER_DIALECT_DEVICE_TRANSFORM_