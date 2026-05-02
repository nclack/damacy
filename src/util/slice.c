#include "util/slice.h"

size_t
cslice_len(struct cslice s)
{
  return (size_t)(s.end - s.beg);
}
