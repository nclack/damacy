#include "dtype.h"

#include <string.h>

size_t
dtype_bpe(enum dtype dt)
{
  switch (dt) {
    case dtype_u8:
    case dtype_i8:
      return 1;
    case dtype_u16:
    case dtype_i16:
    case dtype_f16:
      return 2;
    case dtype_u32:
    case dtype_i32:
    case dtype_f32:
      return 4;
    case dtype_u64:
    case dtype_i64:
    case dtype_f64:
      return 8;
  }
  return 0;
}

int
dtype_from_zarr_string(const char* s, size_t n, enum dtype* out)
{
  static const struct
  {
    const char* name;
    enum dtype dt;
  } table[] = {
    { "uint8", dtype_u8 },    { "uint16", dtype_u16 },
    { "uint32", dtype_u32 },  { "uint64", dtype_u64 },
    { "int8", dtype_i8 },     { "int16", dtype_i16 },
    { "int32", dtype_i32 },   { "int64", dtype_i64 },
    { "float16", dtype_f16 }, { "float32", dtype_f32 },
    { "float64", dtype_f64 },
  };
  for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); ++i) {
    size_t L = strlen(table[i].name);
    if (L == n && memcmp(s, table[i].name, L) == 0) {
      *out = table[i].dt;
      return 0;
    }
  }
  return -1;
}
