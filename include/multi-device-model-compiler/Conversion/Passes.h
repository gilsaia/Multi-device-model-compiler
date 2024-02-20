#ifndef MULTI_DEVICE_MODEL_COMPILER_CONVERSION_PASSES_H_
#define MULTI_DEVICE_MODEL_COMPILER_CONVERSION_PASSES_H_

#include "multi-device-model-compiler/Conversion/ConvertDeviceToLLVM/ConvertDeviceToLLVM.h"
#include "multi-device-model-compiler/Conversion/ConvertDeviceToTPU/ConvertDeviceToTPU.h"
#include "multi-device-model-compiler/Conversion/ConvertGPUToNVVM/ConvertGPUToNVVM.h"
#include "multi-device-model-compiler/Conversion/ConvertMemrefToGPU/ConvertMemrefToGPU.h"
#include "multi-device-model-compiler/Conversion/ConvertONNXToTosa/ConvertONNXToTosa.h"
#include "multi-device-model-compiler/Conversion/ConvertTosaToDevice/ConvertTosaToDevice.h"
#include "multi-device-model-compiler/Conversion/ConvertTosaToLinalg/ConvertTosaToLinalgSaveTensor.h"
#include "multi-device-model-compiler/Conversion/ConvertTosaToTPU/ConvertTosaToTPU.h"

namespace multi_device {
#define GEN_PASS_REGISTRATION
#include "multi-device-model-compiler/Conversion/Passes.h.inc"
} // namespace multi_device

#endif