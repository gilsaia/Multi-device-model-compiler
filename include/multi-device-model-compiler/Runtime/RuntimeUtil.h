#ifndef MULTI_DEVICE_MODEL_COMPILER_RUNTIME_RUNTIMEUTIL_H_
#define MULTI_DEVICE_MODEL_COMPILER_RUNTIME_RUNTIMEUTIL_H_

#include "multi-device-model-compiler/Runtime/ModelInfo.h"
#include "multi-device-model-compiler/Runtime/TensorDescripter.h"

#include <string>
#include <vector>

namespace multi_device {
typedef void (*RunGraphFuncPtr)(...);
float *LoadFloatData(std::string &file, size_t numElements);
template <size_t N>
void SaveFloatData(std::string &file, FloatTensor<N> *tensor);
template <size_t N>
void *GetFloatTensorDescripter(std::vector<size_t> &shapes, std::string file,
                               bool init, bool useFile);
void SaveFloatTensor(ModelInfo *info, std::vector<void *> params,
                     std::string &file);
std::vector<void *> GetParamsVec(ModelInfo *info);
void *LoadLibrary(std::string libName);
RunGraphFuncPtr LoadFunc(void *handle, std::string funcName);
void RunGraphFunc(RunGraphFuncPtr func, std::vector<void *> &params);
void ClearOutputTensor(ModelInfo *info, std::vector<void *> &params);
} // namespace multi_device

#endif