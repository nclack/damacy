#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // CRC32C (Castagnoli). Self-initializing; thread-safe.
  uint32_t crc32c(const void* data, size_t len);

#ifdef __cplusplus
}
#endif
