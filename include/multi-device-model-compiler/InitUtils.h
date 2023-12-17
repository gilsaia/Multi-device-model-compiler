#ifndef MULTI_DEVICE_MODEL_COMPILER_INITUTILS_H_
#define MULTI_DEVICE_MODEL_COMPILER_INITUTILS_H_

namespace multi_device {
void initONNXPasses();
void initConvertPassPipelines();
void initMultiDevicePasses();
} // namespace multi_device

#endif