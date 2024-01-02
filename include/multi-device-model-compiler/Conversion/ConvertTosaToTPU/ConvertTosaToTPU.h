#ifndef MULTI_DEVICE_MODEL_COMPILER_CONVERSION_CONVERTTOSATOTPU_CONVERTTOSATOTPU_H_
#define MULTI_DEVICE_MODEL_COMPILER_CONVERSION_CONVERTTOSATOTPU_CONVERTTOSATOTPU_H_

#include "mlir/Pass/Pass.h"

namespace multi_device {
std::unique_ptr<mlir::Pass> createConvertTosaToTPUPass();
}

#endif