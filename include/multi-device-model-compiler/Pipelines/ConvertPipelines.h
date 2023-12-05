#ifndef MULTI_DEVICE_MODEL_COMPILER_PIPELINES_CONVERTPIPELINES_H_
#define MULTI_DEVICE_MODEL_COMPILER_PIPELINES_CONVERTPIPELINES_H_

#include "mlir/Pass/PassManager.h"

namespace multi_device {
namespace pipelines {
void createONNXToMLIRPipeline(mlir::OpPassManager &pm);
void createMLIRToCPUPipeline(mlir::OpPassManager &pm);
void createMLIRToGPUPipeline(mlir::OpPassManager &pm);
void createMLIRToTPUPipeline(mlir::OpPassManager &pm);
} // namespace pipelines
} // namespace multi_device

#endif