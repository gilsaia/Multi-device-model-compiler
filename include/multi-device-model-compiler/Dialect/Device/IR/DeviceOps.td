#ifndef MULTI_DEVICE_MODEL_COMPILER_DIALECT_DEVICE_OPS_
#define MULTI_DEVICE_MODEL_COMPILER_DIALECT_DEVICE_OPS_

include "multi-device-model-compiler/Dialect/Device/IR/DeviceBase.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/Interfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/GPU/IR/GPUBase.td"

class Device_Op<string mnemonic,list<Trait> traits=[]> :
    Op<Device_Dialect,mnemonic,traits>;

def Device_WaitOp : Device_Op<"wait",[GPU_AsyncOpInterface,Symbol]> {
    let summary = "Wait for async ops to compute.";
    let description = [{
        This op sync the target device with a list of dependent ops.

        In GPU, it will try to generate event to optimize performance.
    }];

    let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies);
    let results = (outs Optional<GPU_AsyncToken>:$asyncToken);

    let assemblyFormat = [{
        custom<AsyncDependencies>(type($asyncToken), $asyncDependencies) attr-dict
    }];
}

def Device_RecordOp : Device_Op<"record"> {
    let summary = "Record dependency for stream use event.";
    let description = [{
        This op used to record dependency within different streams, 
        will translate to event record.
    }];

    let arguments = (ins 
    GPU_AsyncToken:$src,
    GPU_AsyncToken:$dst);

    let assemblyFormat = [{
        `async` ` ` `[` $src `->` $dst `]` attr-dict
    }];
}

def Device_MatmulOp : Device_Op<"matmul",[
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<GPU_AsyncOpInterface,["addAsyncDependency"]>
    ]> {
    let summary = "Matmul op for device to different func call.";
    let description = [{
        Use different methods to accerlate compute.
    }];

    let arguments = (ins
        AnyShaped:$input,
        AnyShaped:$weight,
        AnyShaped:$bias,
        AnyShaped:$output,
        Variadic<GPU_AsyncToken>:$asyncDependencies);

    let results = (outs Optional<GPU_AsyncToken>:$asyncToken);

    let hasVerifier = 1;
    let assemblyFormat = [{
        custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
        `{` 
        `input` $input `:` type($input)
        `weight` $weight `:` type($weight) 
        `bias` $bias `:` type($bias)
        `output` $output `:` type($output)
        `}` attr-dict
    }];
}

def Device_Conv2DOp : Device_Op<"conv2d",[
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<GPU_AsyncOpInterface,["addAsyncDependency"]>,
    AttrSizedOperandSegments 
]> {
    let summary = "performs a 2D convolution over given tensor input";

    let arguments = (ins
        AnyShaped:$input,
        AnyShaped:$weight,
        AnyShaped:$bias,
        AnyShaped:$output,
        Optional<AnyShaped>:$postadd,
        Variadic<GPU_AsyncToken>:$asyncDependencies,

        DenseI64ArrayAttr:$kernel,
        DenseI64ArrayAttr:$pad,
        DenseI64ArrayAttr:$stride,
        DenseI64ArrayAttr:$dilation,
        BoolAttr:$containRelu
    );

    let results = (outs Optional<GPU_AsyncToken>:$asyncToken);

    let hasVerifier = 1;
    let assemblyFormat = [{
        custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
        `{` 
        `input` $input `:` type($input)
        `weight` $weight `:` type($weight) 
        `bias` $bias `:` type($bias)
        `output` $output `:` type($output)
        (`postadd` $postadd^ `:` type($postadd))?
        `}` attr-dict
    }];
}

def Device_Pool2DOp : Device_Op<"pool2d",[
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<GPU_AsyncOpInterface,["addAsyncDependency"]>,
]> {
    let summary = "performs a 2D convolution over given tensor input";

    let arguments = (ins
        AnyShaped:$input,
        AnyShaped:$output,
        Variadic<GPU_AsyncToken>:$asyncDependencies,

        DenseI64ArrayAttr:$kernel,
        DenseI64ArrayAttr:$pad,
        DenseI64ArrayAttr:$stride,
        StrAttr:$method
    );

    let results = (outs Optional<GPU_AsyncToken>:$asyncToken);

    let hasVerifier = 1;
    let assemblyFormat = [{
        custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
        `{` 
        `input` $input `:` type($input)
        `output` $output `:` type($output)
        `}` attr-dict
    }];
}

#endif // MULTI_DEVICE_MODEL_COMPILER_DIALECT_DEVICE_OPS_