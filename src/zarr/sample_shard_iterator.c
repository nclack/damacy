#include "zarr/sample_shard_iterator.h"

#include "log/log.h"
#include "util/prelude.h"

#include <string.h>

int
sample_shard_iterator_init(struct sample_shard_iterator* it,
                           const struct zarr_metadata* meta,
                           const struct damacy_aabb* aabb)
{
  CHECK(Bad, it);
  CHECK(Bad, meta);
  CHECK(Bad, aabb);
  memset(it, 0, sizeof(*it));
  CHECK(Bad, aabb->rank == meta->rank);
  CHECK(Bad, meta->rank > 0);
  CHECK(Bad, meta->rank <= DAMACY_MAX_RANK);

  it->rank = meta->rank;
  for (uint8_t d = 0; d < meta->rank; ++d) {
    uint64_t chunk_extent = meta->inner_chunk_shape[d];
    uint64_t shard_extent = meta->shard_shape[d];
    CHECK(Bad, chunk_extent > 0);
    CHECK(Bad, shard_extent > 0);
    CHECK(Bad, shard_extent % chunk_extent == 0);
    uint64_t inner_per_shard = shard_extent / chunk_extent;

    int64_t beg = aabb->dims[d].beg;
    int64_t end = aabb->dims[d].end;
    CHECK(Bad, beg >= 0);
    CHECK(Bad, end > beg);
    CHECK(Bad, (uint64_t)end <= meta->shape[d]);

    uint64_t chunk_beg = (uint64_t)beg / chunk_extent;
    uint64_t chunk_end = ((uint64_t)end - 1) / chunk_extent + 1;
    it->shard_beg[d] = chunk_beg / inner_per_shard;
    it->shard_end[d] = (chunk_end - 1) / inner_per_shard + 1;
    it->cursor[d] = it->shard_beg[d];
  }
  return 0;
Bad:
  return 1;
}

int
sample_shard_iterator_next(struct sample_shard_iterator* it,
                           uint64_t out[DAMACY_MAX_RANK])
{
  CHECK(End, it);
  CHECK(End, out);
  if (it->finished)
    return 0;

  for (uint8_t d = 0; d < it->rank; ++d)
    out[d] = it->cursor[d];

  int8_t d = (int8_t)it->rank - 1;
  while (d >= 0) {
    it->cursor[d]++;
    if (it->cursor[d] < it->shard_end[d])
      return 1;
    it->cursor[d] = it->shard_beg[d];
    d--;
  }
  it->finished = 1;
  return 1;
End:
  return 0;
}
