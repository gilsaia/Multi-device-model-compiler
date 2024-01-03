#include "multi-device-model-compiler/Conversion/ConvertTosaToTPU/ConvertTosaToTPU.h"
#include "multi-device-model-compiler/Dialect/Device/IR/Device.h"
#include "multi-device-model-compiler/Pass/InitPasses.h"
#include "multi-device-model-compiler/Pipelines/ConvertPipelines.h"

void multi_device::pipelines::createMLIRToTPUPipeline(mlir::OpPassManager &pm) {
  device::AddDeviceTypeToFuncOptions deviceOptions;
  deviceOptions.deviceType = device::DeviceType::TPU;
  pm.addPass(
      multi_device::device::createAddDeviceTypeToFuncPass(deviceOptions));
  pm.addPass(multi_device::createEliminateEntryPointPass());
  pm.addPass(multi_device::createConvertTosaToTPUPass());
}