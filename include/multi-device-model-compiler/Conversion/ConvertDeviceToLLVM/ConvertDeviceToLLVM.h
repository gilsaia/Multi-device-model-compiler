#ifndef MULTI_DEVICE_MODEL_COMPILER_CONVERSION_CONVERTDEVICETOLLVM_H_
#define MULTI_DEVICE_MODEL_COMPILER_CONVERSION_CONVERTDEVICETOLLVM_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Utils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace multi_device {
#define GEN_PASS_DECL_CONVERTDEVICETOLLVM
#include "multi-device-model-compiler/Conversion/Passes.h.inc"
namespace conversion {
void populateDeviceToLLVMConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns,
    mlir::SymbolTable *cachedModuleTable = nullptr);
}
} // namespace multi_device

#endif