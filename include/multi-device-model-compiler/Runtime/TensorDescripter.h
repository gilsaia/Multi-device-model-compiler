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
  static TensorDescriptor *CreateTensor(std::vector<int> &sizes,
                                        std::vector<int> &strides);
};

template <typename T, size_t N>
TensorDescriptor<T, N> *
TensorDescriptor<T, N>::CreateTensor(std::vector<int> &sizes,
                                     std::vector<int> &strides) {
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

template <size_t N> using FloatTensor = TensorDescriptor<float, N>;
using Float1DTensor = FloatTensor<1>;
using Float2DTensor = FloatTensor<2>;
using Float3DTensor = FloatTensor<3>;
using FLoat4DTensor = FloatTensor<4>;

template class TensorDescriptor<float, 1>;
template class TensorDescriptor<float, 2>;
template class TensorDescriptor<float, 3>;
template class TensorDescriptor<float, 4>;

#endif