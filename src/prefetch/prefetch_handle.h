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

#define PREFETCH_HANDLE_NONE ((struct prefetch_handle){ 0, 0 })

#ifdef __cplusplus
}
#endif
