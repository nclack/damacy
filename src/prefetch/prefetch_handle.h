#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct prefetch_handle
  {
    uint32_t slot;
    uint32_t generation;
  };

  static const struct prefetch_handle PREFETCH_HANDLE_NONE = { 0, 0 };

#ifdef __cplusplus
}
#endif
