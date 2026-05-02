// Lightweight read-only string view: a half-open [beg, end) range of bytes.
// cslice borrows its storage; the producer determines lifetime. Reserved
// names: `slice` for a future mutable variant, `slice_<T>` / `slice_c<T>`
// for typed mutable / const variants over other element types.
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct cslice
  {
    const char* beg;
    const char* end;
  };

  size_t cslice_len(struct cslice s);

#ifdef __cplusplus
}
#endif
