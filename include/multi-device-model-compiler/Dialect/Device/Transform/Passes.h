#ifndef MULTI_DEVICE_MODEL_COMPILER_DIALECT_DEVICE_TRANSFORM_PASSES_H_
#define MULTI_DEVICE_MODEL_COMPILER_DIALECT_DEVICE_TRANSFORM_PASSES_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "multi-device-model-compiler/Dialect/Device/IR/Device.h"

namespace multi_device {
namespace device {
#define GEN_PASS_DECL
#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h.inc"

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAddDeviceTypeToFuncPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAddDeviceTypeToFuncPass(const AddDeviceTypeToFuncOptions &type);

void registerBufferizableOpInterfaceExternalModels(
    mlir::DialectRegistry &registry);

#define GEN_PASS_REGISTRATION
#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h.inc"
} // namespace device
} // namespace multi_device

#endif