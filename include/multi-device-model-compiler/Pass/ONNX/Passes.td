#ifndef MULTI_DEVICE_MODEL_COMPILER_ONNX_PASSES
#define MULTI_DEVICE_MODEL_COMPILER_ONNX_PASSES

include "mlir/Pass/PassBase.td"

def EliminateEntryPoint: Pass<"onnx-eliminate-entrypoint">{
    let summary= "Just Eliminate onnx.entrypoiny in module.";
    let constructor="mlir::multi_device::createEliminateEntryPointPass()";
    let dependentDialects=["ONNXDialect"];
}

#endif