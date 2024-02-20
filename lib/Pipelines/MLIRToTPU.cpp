#include "multi-device-model-compiler/Conversion/ConvertDeviceToTPU/ConvertDeviceToTPU.h"
#include "multi-device-model-compiler/Conversion/ConvertTosaToDevice/ConvertTosaToDevice.h"
#include "multi-device-model-compiler/Conversion/ConvertTosaToTPU/ConvertTosaToTPU.h"
#include "multi-device-model-compiler/Dialect/Device/IR/Device.h"
#include "multi-device-model-compiler/Pass/InitPasses.h"
#include "multi-device-model-compiler/Pipelines/ConvertPipelines.h"

#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"

#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

void multi_device::pipelines::createMLIRToTPUPipeline(mlir::OpPassManager &pm) {
  device::AddDeviceTypeToFuncOptions deviceOptions;
  deviceOptions.deviceType = device::DeviceType::TPU;
  pm.addPass(
      multi_device::device::createAddDeviceTypeToFuncPass(deviceOptions));
  pm.addPass(multi_device::createEliminateEntryPointPass());
  auto tosaToDeviceConfig = multi_device::TosaLowerToDeviceOptions();
  tosaToDeviceConfig.useLinalgConvert = false;
  pm.addPass(multi_device::createTosaLowerToDevice(tosaToDeviceConfig));
  pm.addPass(mlir::tosa::createTosaLayerwiseConstantFoldPass());
  pm.addPass(multi_device::createTosaLowerToTPU());
  pm.addPass(multi_device::createDeviceLowerToTPU());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(top::createInitPass());
  pm.addPass(tpu::createStripIOQuant());
  pm.addPass(tpu::createChipOptimizePass());
  pm.addPass(tpu::createDistributePass());
  pm.addPass(tpu::createWeightReorderPass());
  pm.addPass(tpu::createOpDividePass());
  pm.addPass(tpu::createSubnetDividePass());
  pm.addPass(tpu::createOpReorderPass());
  pm.addPass(tpu::createLayerGroupPass());
  pm.addPass(tpu::createParallelPass());
  pm.addPass(tpu::createAddressAssignPass());
  pm.addPass(top::createDeinitPass());
}