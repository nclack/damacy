#include "damacy.h"
#include "expect.h"
#include "zarr/sample_shard_iterator.h"
#include "zarr/zarr_metadata.h"

#include <stdint.h>
#include <string.h>

static struct zarr_metadata
meta_2d(uint64_t shape0,
        uint64_t shape1,
        uint64_t inner0,
        uint64_t inner1,
        uint64_t shard0,
        uint64_t shard1)
{
  struct zarr_metadata m = { .rank = 2 };
  m.shape[0] = shape0;
  m.shape[1] = shape1;
  m.inner_chunk_shape[0] = inner0;
  m.inner_chunk_shape[1] = inner1;
  m.shard_shape[0] = shard0;
  m.shard_shape[1] = shard1;
  return m;
}

static struct damacy_aabb
aabb_2d(int64_t lo0, int64_t hi0, int64_t lo1, int64_t hi1)
{
  struct damacy_aabb a = { .rank = 2 };
  a.dims[0] = (struct damacy_interval){ .beg = lo0, .end = hi0 };
  a.dims[1] = (struct damacy_interval){ .beg = lo1, .end = hi1 };
  return a;
}

static int
test_full_array_yields_full_shard_grid(void)
{
  struct zarr_metadata m = meta_2d(64, 64, 8, 8, 16, 16);
  struct damacy_aabb a = aabb_2d(0, 64, 0, 64);

  struct sample_shard_iterator it;
  EXPECT(sample_shard_iterator_init(&it, &m, &a) == 0);

  uint64_t coord[DAMACY_MAX_RANK];
  uint32_t n = 0;
  uint64_t expected[16][2] = { { 0, 0 }, { 0, 1 }, { 0, 2 }, { 0, 3 },
                               { 1, 0 }, { 1, 1 }, { 1, 2 }, { 1, 3 },
                               { 2, 0 }, { 2, 1 }, { 2, 2 }, { 2, 3 },
                               { 3, 0 }, { 3, 1 }, { 3, 2 }, { 3, 3 } };
  while (sample_shard_iterator_next(&it, coord)) {
    EXPECT(n < 16);
    EXPECT(coord[0] == expected[n][0]);
    EXPECT(coord[1] == expected[n][1]);
    n++;
  }
  EXPECT(n == 16);
  return 0;
}

static int
test_aabb_in_one_shard_yields_one_coord(void)
{
  struct zarr_metadata m = meta_2d(64, 64, 8, 8, 16, 16);
  struct damacy_aabb a = aabb_2d(0, 8, 0, 8);

  struct sample_shard_iterator it;
  EXPECT(sample_shard_iterator_init(&it, &m, &a) == 0);

  uint64_t coord[DAMACY_MAX_RANK];
  EXPECT(sample_shard_iterator_next(&it, coord));
  EXPECT(coord[0] == 0 && coord[1] == 0);
  EXPECT(sample_shard_iterator_next(&it, coord) == 0);
  return 0;
}

static int
test_aabb_spans_shards_in_one_axis(void)
{
  struct zarr_metadata m = meta_2d(64, 64, 8, 8, 16, 16);
  struct damacy_aabb a = aabb_2d(0, 24, 0, 8);

  struct sample_shard_iterator it;
  EXPECT(sample_shard_iterator_init(&it, &m, &a) == 0);

  uint64_t coord[DAMACY_MAX_RANK];
  EXPECT(sample_shard_iterator_next(&it, coord));
  EXPECT(coord[0] == 0 && coord[1] == 0);
  EXPECT(sample_shard_iterator_next(&it, coord));
  EXPECT(coord[0] == 1 && coord[1] == 0);
  EXPECT(sample_shard_iterator_next(&it, coord) == 0);
  return 0;
}

static int
test_partial_chunk_overlap_still_counts_one_shard(void)
{
  struct zarr_metadata m = meta_2d(64, 64, 8, 8, 16, 16);
  struct damacy_aabb a = aabb_2d(4, 12, 4, 12);

  struct sample_shard_iterator it;
  EXPECT(sample_shard_iterator_init(&it, &m, &a) == 0);
  uint64_t coord[DAMACY_MAX_RANK];
  EXPECT(sample_shard_iterator_next(&it, coord));
  EXPECT(coord[0] == 0 && coord[1] == 0);
  EXPECT(sample_shard_iterator_next(&it, coord) == 0);
  return 0;
}

static int
test_non_sharded_one_chunk_per_shard(void)
{
  struct zarr_metadata m = meta_2d(64, 64, 8, 8, 8, 8);
  struct damacy_aabb a = aabb_2d(0, 16, 0, 16);

  struct sample_shard_iterator it;
  EXPECT(sample_shard_iterator_init(&it, &m, &a) == 0);
  uint32_t n = 0;
  uint64_t coord[DAMACY_MAX_RANK];
  while (sample_shard_iterator_next(&it, coord))
    n++;
  EXPECT(n == 4);
  return 0;
}

static int
test_rank_1(void)
{
  struct zarr_metadata m = { .rank = 1 };
  m.shape[0] = 100;
  m.inner_chunk_shape[0] = 10;
  m.shard_shape[0] = 20;

  struct damacy_aabb a = { .rank = 1 };
  a.dims[0] = (struct damacy_interval){ .beg = 5, .end = 45 };

  struct sample_shard_iterator it;
  EXPECT(sample_shard_iterator_init(&it, &m, &a) == 0);
  uint32_t n = 0;
  uint64_t coord[DAMACY_MAX_RANK];
  uint64_t expected[] = { 0, 1, 2 };
  while (sample_shard_iterator_next(&it, coord)) {
    EXPECT(coord[0] == expected[n]);
    n++;
  }
  EXPECT(n == 3);
  return 0;
}

static int
test_bad_inputs(void)
{
  struct sample_shard_iterator it;
  struct zarr_metadata m = meta_2d(64, 64, 8, 8, 16, 16);
  struct damacy_aabb a = aabb_2d(0, 8, 0, 8);

  EXPECT(sample_shard_iterator_init(NULL, &m, &a) != 0);
  EXPECT(sample_shard_iterator_init(&it, NULL, &a) != 0);
  EXPECT(sample_shard_iterator_init(&it, &m, NULL) != 0);

  struct damacy_aabb bad = a;
  bad.rank = 3;
  EXPECT(sample_shard_iterator_init(&it, &m, &bad) != 0);

  bad = a;
  bad.dims[1].end = 1000;
  EXPECT(sample_shard_iterator_init(&it, &m, &bad) != 0);

  bad = a;
  bad.dims[1].beg = -1;
  EXPECT(sample_shard_iterator_init(&it, &m, &bad) != 0);

  bad = a;
  bad.dims[0].beg = 10;
  bad.dims[0].end = 10;
  EXPECT(sample_shard_iterator_init(&it, &m, &bad) != 0);

  struct zarr_metadata bad_meta = meta_2d(64, 64, 8, 8, 16, 15);
  EXPECT(sample_shard_iterator_init(&it, &bad_meta, &a) != 0);

  return 0;
}

int
main(void)
{
  RUN(test_full_array_yields_full_shard_grid);
  RUN(test_aabb_in_one_shard_yields_one_coord);
  RUN(test_aabb_spans_shards_in_one_axis);
  RUN(test_partial_chunk_overlap_still_counts_one_shard);
  RUN(test_non_sharded_one_chunk_per_shard);
  RUN(test_rank_1);
  RUN(test_bad_inputs);
  log_info("all tests passed");
  return 0;
}
