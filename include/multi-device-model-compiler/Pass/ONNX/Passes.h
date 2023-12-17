#ifndef MULTI_DEVICE_MODEL_COMPILER_PASS_ONNX_H_
#define MULTI_DEVICE_MODEL_COMPILER_PASS_ONNX_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class ONNXDialect;
}

namespace multi_device {
#define GEN_PASS_DECL
#include "multi-device-model-compiler/Pass/ONNX/Passes.h.inc"

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createEliminateEntryPointPass();

#define GEN_PASS_REGISTRATION
#include "multi-device-model-compiler/Pass/ONNX/Passes.h.inc"
} // namespace multi_device

#endif