#ifndef MULTI_DEVICE_MODEL_COMPILER_RUNTIME_TPU_TPURUNTIMEWRAPPERS_H_
#define MULTI_DEVICE_MODEL_COMPILER_RUNTIME_TPU_TPURUNTIMEWRAPPERS_H_

#include "multi-device-model-compiler/Runtime/ModelInfo.h"

#include "bmlib_runtime.h"
#include "bmruntime_interface.h"

#include <string>

bm_handle_t InitHandle();
void *InitRuntime(bm_handle_t handle);
void DestroyRuntime(bm_handle_t handle, void *runtime);
bool LoadBModel(void *runtime, std::string model);
std::vector<void *> GetTensorData(multi_device::ModelInfo *info);
void SaveTensorData(multi_device::ModelInfo *info, std::vector<void *> params,
                    std::string &file);
std::string GetNetName(void *runtime);
void LaunchModel(multi_device::ModelInfo *info, void *runtime,
                 std::vector<void *> tensors, std::string net);

void Sync(bm_handle_t handle);

#endif