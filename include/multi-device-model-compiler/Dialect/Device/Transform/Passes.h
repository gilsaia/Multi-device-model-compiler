#ifndef MULTI_DEVICE_MODEL_COMPILER_DIALECT_DEVICE_TRANSFORM_PASSES_H_
#define MULTI_DEVICE_MODEL_COMPILER_DIALECT_DEVICE_TRANSFORM_PASSES_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include "multi-device-model-compiler/Dialect/Device/IR/Device.h"

namespace multi_device {
namespace device {
#define GEN_PASS_DECL
#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h.inc"

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createAddDeviceTypeToFuncPass();
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createAddDeviceTypeToFuncPass(const AddDeviceTypeToFuncOptions &type);

#define GEN_PASS_REGISTRATION
#include "multi-device-model-compiler/Dialect/Device/Transform/Passes.h.inc"
} // namespace device
} // namespace multi_device

#endif