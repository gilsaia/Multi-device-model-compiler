#ifndef MULTI_DEVICE_COMPILER_KERNELS_GPU_CUDATYPEUTILS_CUH_
#define MULTI_DEVICE_COMPILER_KERNELS_GPU_CUDATYPEUTILS_CUH_

#include "cuda_fp16.h"

template <typename T> struct num_elems;
template <> struct num_elems<float> {
  static constexpr int value = 1;
};
template <> struct num_elems<float2> {
  static constexpr int value = 2;
};
template <> struct num_elems<float4> {
  static constexpr int value = 4;
};
template <> struct num_elems<half> {
  static constexpr int value = 1;
};
template <> struct num_elems<half2> {
  static constexpr int value = 2;
};

template <typename T, int num> struct packed_as;
template <typename T> struct packed_as<T, 1> {
  using type = T;
};
template <> struct packed_as<half, 2> {
  using type = half2;
};
template <> struct packed_as<float, 2> {
  using type = float2;
};
template <> struct packed_as<int8_t, 2> {
  using type = int16_t;
};
template <> struct packed_as<int32_t, 2> {
  using type = int2;
};
template <> struct packed_as<half2, 1> {
  using type = half;
};

inline __device__ float2 operator*(float2 a, float2 b) {
  return make_float2(a.x * b.x, a.y * b.y);
}
inline __device__ float2 operator*(float2 a, float b) {
  return make_float2(a.x * b, a.y * b);
}

template <typename T> inline __device__ T ldg(const T *val) {
  return __ldg(val);
}

// Get type2 from type or vice versa (applied to half and bfloat16)
template <typename T> struct TypeConverter {
  using Type = half2;
}; // keep for generality

template <> struct TypeConverter<half2> {
  using Type = half;
};

template <> struct TypeConverter<half> {
  using Type = half2;
};

// Defined math operations (bfloat16 fallback to fp32 when it is not supported)
template <typename T> inline __device__ T hadd2(T a, T b) {
  return __hadd2(a, b);
}

template <typename T> inline __device__ T add(T a, T b) { return a + b; }

template <> inline __device__ half2 add(half2 a, half2 b) {
  return __hadd2(a, b);
}

template <> inline __device__ half add(half a, half b) { return __hadd(a, b); }

template <typename T> inline __device__ T hmul2(T a, T b) {
  return __hmul2(a, b);
}

template <typename T> inline __device__ T hmul2(T a, T b, T c) {
  return a * b * c;
}

template <typename T> inline __device__ T hsub2(T a, T b) {
  return __hsub2(a, b);
}

template <typename T_OUT, typename T_IN>
__device__ inline T_OUT cuda_cast(T_IN val) {
  return val;
}

template <> __device__ inline float2 cuda_cast<float2, int2>(int2 val) {
  return make_float2(val.x, val.y);
}
template <> __device__ inline float2 cuda_cast<float2, float>(float val) {
  return make_float2(val, val);
}
template <> __device__ inline float2 cuda_cast<float2, half2>(half2 val) {
  return __half22float2(val);
}
template <> __device__ inline half2 cuda_cast<half2, float2>(float2 val) {
  return __float22half2_rn(val);
}
template <> __device__ inline half2 cuda_cast<half2, float>(float val) {
  return __float2half2_rn(val);
}
template <> __device__ inline half2 cuda_cast<half2, half>(half val) {
  return __half2half2(val);
}

template <> __device__ inline int8_t cuda_cast<int8_t, half>(half val) {
  union {
    int8_t int8[2];
    int16_t int16;
  };
  union {
    half fp16;
    int16_t int16_in;
  };
  fp16 = val;
  asm volatile("cvt.rni.sat.s8.f16 %0, %1;" : "=h"(int16) : "h"(int16_in));
  return int8[0];
}

template <> __device__ inline int16_t cuda_cast<int16_t, half2>(half2 val) {
  union {
    int8_t int8[2];
    int16_t int16;
  };
  int8[0] = cuda_cast<int8_t>(val.x);
  int8[1] = cuda_cast<int8_t>(val.y);
  return int16;
}

template <> __device__ inline int8_t cuda_cast<int8_t, float>(float val) {
  union {
    int8_t int8[2];
    int16_t int16;
  };
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=h"(int16) : "f"(val));
  return int8[0];
}

template <> __device__ inline int16_t cuda_cast<int16_t, float2>(float2 val) {
  union {
    int8_t int8[2];
    int16_t int16;
  };
  int8[0] = cuda_cast<int8_t>(val.x);
  int8[1] = cuda_cast<int8_t>(val.y);
  return int16;
}

template <> __device__ inline half2 cuda_cast<half2, int16_t>(int16_t val) {
  union {
    int8_t int8[2];
    int16_t int16;
  };
  int16 = val;
  return make_half2(int8[0], int8[1]);
}

template <> __device__ inline float2 cuda_cast<float2, int16_t>(int16_t val) {
  union {
    int8_t int8[2];
    int16_t int16;
  };
  int16 = val;
  return make_float2(int8[0], int8[1]);
}

template <typename T> __device__ inline T cuda_abs(T val);
template <> __device__ inline float cuda_abs(float val) { return fabs(val); }
template <> __device__ inline half cuda_abs(half val) { return __habs(val); }
template <> __device__ inline half2 cuda_abs(half2 val) { return __habs2(val); }

// Unary maximum: compute the max of a vector type
template <typename To, typename Ti> __device__ inline To cuda_max(Ti val) {
  return cuda_cast<To>(val);
};

template <> __device__ inline half cuda_max(half2 val) {
  return (val.x > val.y) ? val.x : val.y;
}
// Binary maximum: compute the max of two scalar types
template <typename T> __device__ inline T cuda_max(T val1, T val2) {
  return (val1 > val2) ? val1 : val2;
}

#endif