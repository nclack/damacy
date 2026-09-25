#pragma once

#include "damacy.h"
#include "dtype/dtype.h"

#include <stdint.h>
#include <string.h>

#ifdef __CUDACC__
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#define DTYPE_INLINE static __host__ __device__ __forceinline__
#else
#define DTYPE_INLINE static inline
#endif

#define DTYPE_LOAD(NAME, TYPE)                                                 \
  DTYPE_INLINE TYPE dtype_load_##NAME(const void* source)                      \
  {                                                                            \
    TYPE value;                                                                \
    memcpy(&value, source, sizeof(value));                                     \
    return value;                                                              \
  }

DTYPE_LOAD(u8, uint8_t)
DTYPE_LOAD(u16, uint16_t)
DTYPE_LOAD(u32, uint32_t)
DTYPE_LOAD(u64, uint64_t)
DTYPE_LOAD(i8, int8_t)
DTYPE_LOAD(i16, int16_t)
DTYPE_LOAD(i32, int32_t)
DTYPE_LOAD(i64, int64_t)
DTYPE_LOAD(f32, float)

#undef DTYPE_LOAD

DTYPE_INLINE float
dtype_half_to_float(uint16_t half)
{
#ifdef __CUDA_ARCH__
  __half value;
  memcpy(&value, &half, sizeof(value));
  return __half2float(value);
#else
  uint32_t sign = (uint32_t)(half & 0x8000) << 16;
  uint32_t exponent = (half >> 10) & 31;
  uint32_t mantissa = half & 1023;
  uint32_t bits;
  if (!exponent) {
    if (!mantissa)
      bits = sign;
    else {
      int shift = 0;
      while (!(mantissa & 1024)) {
        mantissa <<= 1;
        ++shift;
      }
      bits = sign | (uint32_t)(113 - shift) << 23 | (mantissa & 1023) << 13;
    }
  } else {
    bits =
      sign | (exponent == 31 ? 255 : exponent + 112) << 23 | mantissa << 13;
  }
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
#endif
}

DTYPE_INLINE uint16_t
dtype_float_to_bfloat(float value)
{
#ifdef __CUDA_ARCH__
  __nv_bfloat16 result = __float2bfloat16(value);
  uint16_t bits;
  memcpy(&bits, &result, sizeof(bits));
  return bits;
#else
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7fffffff) > 0x7f800000)
    return 0x7fff;
  return (uint16_t)((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
#endif
}

DTYPE_INLINE uint64_t
dtype_load_unsigned(const void* source, enum dtype type)
{
  switch (type) {
    case dtype_u8:
      return dtype_load_u8(source);
    case dtype_u16:
      return dtype_load_u16(source);
    case dtype_u32:
      return dtype_load_u32(source);
    default:
      return dtype_load_u64(source);
  }
}

DTYPE_INLINE int64_t
dtype_load_signed(const void* source, enum dtype type)
{
  switch (type) {
    case dtype_i8:
      return dtype_load_i8(source);
    case dtype_i16:
      return dtype_load_i16(source);
    case dtype_i32:
      return dtype_load_i32(source);
    default:
      return dtype_load_i64(source);
  }
}

DTYPE_INLINE float
dtype_as_float(const void* source, enum dtype type)
{
  switch (type) {
    case dtype_u8:
      return (float)dtype_load_u8(source);
    case dtype_u16:
      return (float)dtype_load_u16(source);
    case dtype_u32:
      return (float)dtype_load_u32(source);
    case dtype_u64:
      return (float)dtype_load_u64(source);
    case dtype_i8:
      return (float)dtype_load_i8(source);
    case dtype_i16:
      return (float)dtype_load_i16(source);
    case dtype_i32:
      return (float)dtype_load_i32(source);
    case dtype_i64:
      return (float)dtype_load_i64(source);
    case dtype_f16:
      return dtype_half_to_float(dtype_load_u16(source));
    default:
      return dtype_load_f32(source);
  }
}

DTYPE_INLINE uint64_t
dtype_as_unsigned(const void* source, enum dtype type, uint64_t maximum)
{
  if (type <= dtype_u64) {
    uint64_t value = dtype_load_unsigned(source, type);
    return value > maximum ? maximum : value;
  }
  if (type <= dtype_i64) {
    int64_t value = dtype_load_signed(source, type);
    if (value < 0)
      return 0;
    return (uint64_t)value > maximum ? maximum : (uint64_t)value;
  }
  float value = dtype_as_float(source, type);
  if (!(value > 0))
    return 0;
  if (value >= (float)maximum)
    return maximum;
  return (uint64_t)value;
}

DTYPE_INLINE int64_t
dtype_as_signed(const void* source,
                enum dtype type,
                int64_t minimum,
                int64_t maximum)
{
  if (type <= dtype_u64) {
    uint64_t value = dtype_load_unsigned(source, type);
    return value > (uint64_t)maximum ? maximum : (int64_t)value;
  }
  if (type <= dtype_i64) {
    int64_t value = dtype_load_signed(source, type);
    return value < minimum ? minimum : value > maximum ? maximum : value;
  }
  float value = dtype_as_float(source, type);
  if (value != value)
    return 0;
  if (value <= (float)minimum)
    return minimum;
  if (value >= (float)maximum)
    return maximum;
  return (int64_t)value;
}

DTYPE_INLINE int
dtype_matches(enum damacy_dtype destination, enum dtype source)
{
  switch (destination) {
    case DAMACY_F32:
      return source == dtype_f32;
    case DAMACY_U8:
      return source == dtype_u8;
    case DAMACY_U16:
      return source == dtype_u16;
    case DAMACY_U32:
      return source == dtype_u32;
    case DAMACY_U64:
      return source == dtype_u64;
    case DAMACY_I8:
      return source == dtype_i8;
    case DAMACY_I16:
      return source == dtype_i16;
    case DAMACY_I32:
      return source == dtype_i32;
    case DAMACY_I64:
      return source == dtype_i64;
    default:
      return 0;
  }
}

DTYPE_INLINE void
dtype_convert(void* destination,
              enum damacy_dtype output,
              const void* source,
              enum dtype input)
{
#define DTYPE_STORE(DTYPE, TYPE, VALUE)                                        \
  case DTYPE: {                                                                \
    TYPE value = (TYPE)(VALUE);                                                \
    memcpy(destination, &value, sizeof(value));                                \
    break;                                                                     \
  }

  switch (output) {
    DTYPE_STORE(DAMACY_F32, float, dtype_as_float(source, input))
    DTYPE_STORE(DAMACY_BF16,
                uint16_t,
                dtype_float_to_bfloat(dtype_as_float(source, input)))
    DTYPE_STORE(DAMACY_U8, uint8_t, dtype_as_unsigned(source, input, UINT8_MAX))
    DTYPE_STORE(
      DAMACY_U16, uint16_t, dtype_as_unsigned(source, input, UINT16_MAX))
    DTYPE_STORE(
      DAMACY_U32, uint32_t, dtype_as_unsigned(source, input, UINT32_MAX))
    DTYPE_STORE(
      DAMACY_U64, uint64_t, dtype_as_unsigned(source, input, UINT64_MAX))
    DTYPE_STORE(
      DAMACY_I8, int8_t, dtype_as_signed(source, input, INT8_MIN, INT8_MAX))
    DTYPE_STORE(
      DAMACY_I16, int16_t, dtype_as_signed(source, input, INT16_MIN, INT16_MAX))
    DTYPE_STORE(
      DAMACY_I32, int32_t, dtype_as_signed(source, input, INT32_MIN, INT32_MAX))
    DTYPE_STORE(
      DAMACY_I64, int64_t, dtype_as_signed(source, input, INT64_MIN, INT64_MAX))
  }

#undef DTYPE_STORE
}

#undef DTYPE_INLINE
