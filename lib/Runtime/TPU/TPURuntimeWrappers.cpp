#include "multi-device-model-compiler/Runtime/TPU/TPURuntimeWrappers.h"
#include "multi-device-model-compiler/Runtime/RuntimeUtil.h"

#include "bmruntime_interface.h"

#define TPU_CHECK_ERROR(call)                                                  \
  do {                                                                         \
    bm_status_t ret = call;                                                    \
    if (ret != BM_SUCCESS) {                                                   \
      llvm::errs() << "TPU_CHECK_ERROR_FAILED" << __FILE__ << " " << __LINE__  \
                   << "\n";                                                    \
    }                                                                          \
  } while (0)

bm_handle_t InitHandle() {
  bm_handle_t handle;
  TPU_CHECK_ERROR(bm_dev_request(&handle, 0));
  return handle;
}

void *InitRuntime(bm_handle_t handle) {
  auto runtime = bmrt_create(handle);
  return runtime;
}

void DestroyRuntime(bm_handle_t handle, void *runtime) {
  bmrt_destroy(runtime);
  bm_dev_free(handle);
}

bool LoadBModel(void *runtime, std::string model) {
  return bmrt_load_bmodel(runtime, model.c_str());
}

std::vector<void *> GetTensorData(multi_device::ModelInfo *info) {
  std::vector<void *> datas;
  for (size_t i = 0; i < info->InputNums(); ++i) {
    size_t numElements = info->GetInputNumElements(i);
    if (info->ExistFile()) {
      std::string file = info->GetFile(i);
      datas.emplace_back(multi_device::LoadFloatData(file, numElements));
    } else {
      datas.emplace_back(aligned_alloc(64, numElements * sizeof(float)));
    }
  }
  for (size_t i = 0; i < info->OutputNums(); ++i) {
    size_t numElements = info->GetOutputNumElements(i);
    datas.emplace_back(aligned_alloc(64, numElements * sizeof(float)));
  }
  return datas;
}

void SaveTensorData(multi_device::ModelInfo *info, std::vector<void *> params,
                    std::string &file) {
  std::error_code ec;
  llvm::raw_fd_ostream output(file, ec);
  if (ec) {
    llvm::errs() << "Error open file to write:" << ec.message() << "\n";
    return;
  }
  output.write(reinterpret_cast<char *>(params.data() + info->InputNums()),
               info->GetOutputNumElements(0) * sizeof(float));
  return;
}

std::string GetNetName(void *runtime) {
  const char **net_names = NULL;
  int net_num = bmrt_get_network_number(runtime);
  if (net_num > 1) {
    llvm::outs() << "There is more than 1 net\n";
    for (int i = 0; i < net_num; ++i) {
      llvm::outs() << "Network " << i << ":\t" << net_names[i] << "\n";
    }
  }
  std::string netName(net_names[0]);
  free(net_names);
  return netName;
}

void LaunchModel(multi_device::ModelInfo *info, void *runtime,
                 std::vector<void *> tensors, std::string net) {
  bm_shape_t *input_shapes = new bm_shape_t[info->InputNums()];
  for (size_t i = 0; i < info->InputNums(); ++i) {
    std::vector<int> shapes;
    for (auto &shape : info->GetInputSize(i)) {
      shapes.emplace_back(shape);
    }
    bmrt_shape(&input_shapes[i], shapes.data(), info->GetInputSize(i).size());
  }
  bm_shape_t *output_shapes = new bm_shape_t[info->OutputNums()];
  for (size_t i = 0; i < info->OutputNums(); ++i) {
    std::vector<int> shapes;
    for (auto &shape : info->GetOutputSize(i)) {
      shapes.emplace_back(shape);
    }
    bmrt_shape(&output_shapes[i], shapes.data(), info->GetOutputSize(i).size());
  }
  bool res =
      bmrt_launch_data(runtime, net.c_str(), tensors.data(), input_shapes,
                       info->InputNums(), tensors.data() + info->InputNums(),
                       output_shapes, info->OutputNums(), true);
  if (!res) {
    llvm::errs() << "Launch wrong!\n";
  }
  delete[] input_shapes;
  delete[] output_shapes;
}

void Sync(bm_handle_t handle) { TPU_CHECK_ERROR(bm_thread_sync(handle)); }