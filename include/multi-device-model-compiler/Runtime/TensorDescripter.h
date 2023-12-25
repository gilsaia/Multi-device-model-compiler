#ifndef MULTI_DEVICE_MODEL_COMPILER_RUNTIME_TENSORDESCRIPTER_H_
#define MULTI_DEVICE_MODEL_COMPILER_RUNTIME_TENSORDESCRIPTER_H_

#include <cstddef>
#include <cstdint>
#include <malloc.h>
#include <vector>

template <typename T, size_t N> struct TensorDescriptor {
  T *allocated;
  T *aligned;
  intptr_t offset;
  intptr_t sizes[N];
  intptr_t strides[N];
  static TensorDescriptor *CreateTensor(std::vector<size_t> &sizes);
  void InitData();
};

template <typename T, size_t N>
TensorDescriptor<T, N> *
TensorDescriptor<T, N>::CreateTensor(std::vector<size_t> &sizes) {
  TensorDescriptor<T, N> *tensor =
      (TensorDescriptor<T, N> *)malloc(sizeof(TensorDescriptor<T, N>));
  tensor->allocated = nullptr;
  tensor->aligned = nullptr;
  tensor->offset = 0;
  int64_t rank = N;
  for (int64_t i = rank - 1; i >= 0; --i) {
    tensor->sizes[i] = sizes[i];
    if (i == (rank - 1)) {
      tensor->strides[i] = 1;
    } else {
      tensor->strides[i] = tensor->strides[i + 1] * tensor->sizes[i + 1];
    }
  }
  return tensor;
}

template <typename T, size_t N> void TensorDescriptor<T, N>::InitData() {
  size_t num = 1;
  for (size_t i = 0; i < N; ++i) {
    num *= sizes[i];
  }
  T *data = (T *)malloc(num * sizeof(T));
  this->allocated = data;
  this->aligned = data;
}

template <size_t N> using FloatTensor = TensorDescriptor<float, N>;
using Float1DTensor = FloatTensor<1>;
using Float2DTensor = FloatTensor<2>;
using Float3DTensor = FloatTensor<3>;
using FLoat4DTensor = FloatTensor<4>;

#endif