#ifndef MULTI_DEVICE_MODEL_COMPILER_CONVERSION_CONVERTONNXTOTOSA_CONVERTONNXTOTOSA_H_
#define MULTI_DEVICE_MODEL_COMPILER_CONVERSION_CONVERTONNXTOTOSA_CONVERTONNXTOTOSA_H_

#include "mlir/Pass/Pass.h"

namespace multi_device {
#define GEN_PASS_DECL_FRONTENDTOTOSALOWERINGFIX
#include "multi-device-model-compiler/Conversion/Passes.h.inc"
} // namespace multi_device

#endif