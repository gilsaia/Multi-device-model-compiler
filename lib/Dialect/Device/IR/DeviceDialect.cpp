#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"

#include "multi-device-model-compiler/Dialect/Device/IR/Device.h"

using namespace multi_device;
using namespace multi_device::device;

#include "multi-device-model-compiler/Dialect/Device/IR/DeviceOpsDialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "multi-device-model-compiler/Dialect/Device/IR/DeviceOpsAttributes.cpp.inc"

void device::DeviceDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "multi-device-model-compiler/Dialect/Device/IR/DeviceOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "multi-device-model-compiler/Dialect/Device/IR/DeviceOpsAttributes.cpp.inc"
      >();
}

#include "multi-device-model-compiler/Dialect/Device/IR/DeviceOpsEnums.cpp.inc"