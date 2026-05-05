#include "dimension/dimension.h"

#include <stdlib.h>
#include <string.h>

int
dims_copy(struct dimension* dst, const struct dimension* src, uint8_t rank)
{
  if (!dst || !src)
    return 1;
  for (uint8_t d = 0; d < rank; ++d) {
    dst[d] = src[d];
    dst[d].name = NULL;
    if (src[d].name) {
      size_t n = strlen(src[d].name);
      char* dup = (char*)malloc(n + 1);
      if (!dup) {
        dims_free_names(dst, d);
        for (uint8_t z = d; z < rank; ++z)
          dst[z] = (struct dimension){ 0 };
        return 1;
      }
      memcpy(dup, src[d].name, n + 1);
      dst[d].name = dup;
    }
  }
  return 0;
}

void
dims_free_names(struct dimension* dims, uint8_t rank)
{
  if (!dims)
    return;
  for (uint8_t d = 0; d < rank; ++d) {
    free((char*)dims[d].name);
    dims[d].name = NULL;
  }
}

uint64_t
dim_n_chunks(const struct dimension* d)
{
  if (!d || d->chunk_size == 0)
    return 0;
  return (d->size + d->chunk_size - 1) / d->chunk_size;
}

uint64_t
dim_shard_extent(const struct dimension* d)
{
  if (!d)
    return 0;
  uint64_t cps = d->chunks_per_shard ? d->chunks_per_shard : 1;
  return d->chunk_size * cps;
}
