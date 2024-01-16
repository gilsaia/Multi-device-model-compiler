#ifndef MULTI_DEVICE_MODEL_COMPILER_CONVERSION_PASSES_H_
#define MULTI_DEVICE_MODEL_COMPILER_CONVERSION_PASSES_H_

#include "multi-device-model-compiler/Conversion/ConvertGPUToNVVM/ConvertGPUToNVVM.h"

namespace multi_device {
#define GEN_PASS_REGISTRATION
#include "multi-device-model-compiler/Conversion/Passes.h.inc"
} // namespace multi_device

#endif