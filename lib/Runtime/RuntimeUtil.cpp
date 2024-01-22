#include "multi-device-model-compiler/Runtime/RuntimeUtil.h"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <dlfcn.h>

namespace multi_device {
float *LoadFloatData(std::string &file, size_t numElements) {
  auto ebuffer = llvm::MemoryBuffer::getFile(file);
  if (auto ec = ebuffer.getError()) {
    llvm::errs() << "Error reading file: " << ec.message() << "\n";
    return nullptr;
  }
  std::unique_ptr<llvm::MemoryBuffer> &buffer = *ebuffer;
  if (buffer->getBufferSize() < (numElements * sizeof(float))) {
    llvm::errs() << "Wrong File size\n";
    return nullptr;
  }
  float *data = new float[numElements];
  memcpy(data, buffer->getBufferStart(), numElements * sizeof(float));
  return data;
}

template <size_t N>
void SaveFloatData(std::string &file, FloatTensor<N> *tensor) {
  std::error_code ec;
  llvm::raw_fd_ostream output(file, ec);
  if (ec) {
    llvm::errs() << "Error open file to write:" << ec.message() << "\n";
    return;
  }
  output.write(reinterpret_cast<char *>(tensor->allocated),
               tensor->GetNumElements());
  return;
}

template <size_t N>
FloatTensor<N> *initFloatTensorDescipter(std::vector<size_t> &shapes,
                                         std::string file, bool init,
                                         bool useFile) {
  FloatTensor<N> *tensor = FloatTensor<N>::CreateTensor(shapes);
  if (init) {
    if (useFile) {
      float *data = LoadFloatData(file, tensor->GetNumElements());
      tensor->InitData(data);
    } else {
      tensor->InitData();
    }
  }
  return tensor;
}

void SaveFloatTensor(ModelInfo *info, std::vector<void *> params,
                     std::string &file) {
  auto &shapes = info->GetOutputSize(0);
  void *tensor = params[0];
  switch (shapes.size()) {
  case 1:
    SaveFloatData(file, reinterpret_cast<Float1DTensor *>(tensor));
    break;
  case 2:
    SaveFloatData(file, reinterpret_cast<Float2DTensor *>(tensor));
    break;
  case 3:
    SaveFloatData(file, reinterpret_cast<Float3DTensor *>(tensor));
    break;
  case 4:
    SaveFloatData(file, reinterpret_cast<FLoat4DTensor *>(tensor));
    break;
  default:
    llvm::errs() << "Not support save dim greater than 4.\n";
    break;
  }
}

void *GetFloatTensorDescripter(std::vector<size_t> &shapes, std::string file,
                               bool init, bool useFile) {
  void *tensor;
  switch (shapes.size()) {
  case 1:
    tensor = initFloatTensorDescipter<1>(shapes, file, init, useFile);
    break;
  case 2:
    tensor = initFloatTensorDescipter<2>(shapes, file, init, useFile);
    break;
  case 3:
    tensor = initFloatTensorDescipter<3>(shapes, file, init, useFile);
    break;
  case 4:
    tensor = initFloatTensorDescipter<4>(shapes, file, init, useFile);
    break;
  default:
    llvm::errs() << "Not support dim greater than 4.\n";
    break;
  }
  return tensor;
}

std::vector<void *> GetParamsVec(ModelInfo *info) {
  std::vector<void *> params;
  for (size_t i = 0; i < info->OutputNums(); ++i) {
    params.emplace_back(
        GetFloatTensorDescripter(info->GetOutputSize(i), "", false, false));
  }
  for (size_t i = 0; i < info->InputNums(); ++i) {
    params.emplace_back(GetFloatTensorDescripter(
        info->GetInputSize(i), info->ExistFile() ? info->GetFile(i) : "", true,
        info->ExistFile()));
  }
  return params;
}

void *LoadLibrary(std::string libName) {
  void *handle = dlopen(libName.c_str(), RTLD_LAZY);
  if (!handle) {
    llvm::errs() << "Failed to open library:" << dlerror() << "\n";
    return nullptr;
  }
  return handle;
}

RunGraphFuncPtr LoadFunc(void *handle, std::string funcName) {
  RunGraphFuncPtr func = (RunGraphFuncPtr)dlsym(handle, funcName.c_str());
  const char *err = dlerror();
  if (err) {
    llvm::errs() << "Failed to find run function: " << err << "\n";
    return nullptr;
  }
  return func;
}

void RunGraphFunc(RunGraphFuncPtr func, std::vector<void *> &params) {
  switch (params.size()) {
  case 0:
    func();
    break;
  case 1:
    func(params[0]);
    break;
  case 2:
    func(params[0], params[1]);
    break;
  case 3:
    func(params[0], params[1], params[2]);
    break;
  case 4:
    func(params[0], params[1], params[2], params[3]);
    break;
  case 5:
    func(params[0], params[1], params[2], params[3], params[4]);
    break;
  default:
    llvm::errs() << "Not Support params more than 5.\n";
    break;
  }
}

void ClearOutputTensor(ModelInfo *info, std::vector<void *> &params) {
  for (size_t i = 0; i < info->OutputNums(); ++i) {
    switch (info->GetOutputSize(i).size()) {
    case 1:
      reinterpret_cast<Float1DTensor *>(params[i])->clear();
      break;
    case 2:
      reinterpret_cast<Float2DTensor *>(params[i])->clear();
      break;
    case 3:
      reinterpret_cast<Float3DTensor *>(params[i])->clear();
      break;
    case 4:
      reinterpret_cast<FLoat4DTensor *>(params[i])->clear();
      break;
    default:
      llvm::errs() << "Not support clear tensor which dim more than 4.\n";
      break;
    }
  }
}

} // namespace multi_device