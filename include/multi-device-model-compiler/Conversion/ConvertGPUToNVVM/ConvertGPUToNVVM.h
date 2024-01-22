#ifndef MULTI_DEVICE_MODEL_COMPILER_CONVERSION_CONVERTGPUTONVVM_H_
#define MULTI_DEVICE_MODEL_COMPILER_CONVERSION_CONVERTGPUTONVVM_H_

#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Pass/Pass.h"

namespace multi_device {
#define GEN_PASS_DECL_CONVERTGPUOPSTONVVMOPSFIX
#include "multi-device-model-compiler/Conversion/Passes.h.inc"

} // namespace multi_device

#endif