#ifndef MULTI_DEVICE_MODEL_COMPILER_DIALECT_ONNX_TRANSFORM_
#define MULTI_DEVICE_MODEL_COMPILER_DIALECT_ONNX_TRANSFORM_

include "mlir/Pass/PassBase.td"

def EliminateEntryPoint: Pass<"onnx-eliminate-entrypoint","mlir::ModuleOp">{
    let summary= "Just Eliminate onnx.entrypoint in module.";
    let description=[{
        Use model-converter will cause onnx.entrypoint,which is not used in the future,so 
        this pass will eliminate entrypoint.
    }];
    let constructor="multi_device::createEliminateEntryPointPass()";
    let dependentDialects=["mlir::ONNXDialect"];
}
#endif // MULTI_DEVICE_MODEL_COMPILER_DIALECT_ONNX_TRANSFORM_