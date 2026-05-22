#pragma once

#include "damacy.h"
#include "damacy_limits.h"
#include "zarr/zarr_metadata.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct sample_shard_iterator
  {
    uint8_t rank;
    uint64_t shard_beg[DAMACY_MAX_RANK];
    uint64_t shard_end[DAMACY_MAX_RANK];
    uint64_t cursor[DAMACY_MAX_RANK];
    int finished;
  };

  int sample_shard_iterator_init(struct sample_shard_iterator* it,
                                 const struct zarr_metadata* meta,
                                 const struct damacy_aabb* aabb);
  int sample_shard_iterator_next(struct sample_shard_iterator* it,
                                 uint64_t out[DAMACY_MAX_RANK]);

#ifdef __cplusplus
}
#endif
