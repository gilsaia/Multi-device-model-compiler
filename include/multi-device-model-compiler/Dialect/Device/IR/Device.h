#ifndef MULTI_DEVICE_MODEL_COMPILER_DIALECT_DEVICE_IR_DEVICE_H_
#define MULTI_DEVICE_MODEL_COMPILER_DIALECT_DEVICE_IR_DEVICE_H_

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "llvm/ADT/StringExtras.h"

#include "multi-device-model-compiler/Dialect/Device/IR/DeviceOpsDialect.h.inc"

#include "multi-device-model-compiler/Dialect/Device/IR/DeviceOpsEnums.h.inc"
using namespace mlir;
#define GET_ATTRDEF_CLASSES
#include "multi-device-model-compiler/Dialect/Device/IR/DeviceOpsAttributes.h.inc"
#define GET_OP_CLASSES
#include "multi-device-model-compiler/Dialect/Device/IR/DeviceOps.h.inc"

#endif