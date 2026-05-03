// Integration test for the planner. Spins up a tmpdir-backed fs store
// with a synthetic 2D zarr (one shard, four inner chunks) and verifies
// planner output: read counts, page alignment, src/dst windows.

#include "damacy.h"
#include "fixture.h"
#include "planner.h"
#include "store.h"
#include "zarr_meta_cache.h"
#include "zarr_shard_cache.h"
#include "zarr_shard_index.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define PAGE 4096u

// 2D shape=[4,8], inner=[2,4], shard=[4,8] → one shard, 4 inner chunks
// arranged 2x2 within the shard. uint16 → inner chunk = 2*4*2 = 16 B.
static const char* MINIMAL_ZARR_JSON =
  "{"
  "\"zarr_format\":3,"
  "\"node_type\":\"array\","
  "\"shape\":[4,8],"
  "\"data_type\":\"uint16\","
  "\"chunk_grid\":{\"name\":\"regular\",\"configuration\":{"
  "\"chunk_shape\":[4,8]}},"
  "\"chunk_key_encoding\":{\"name\":\"default\",\"configuration\":{"
  "\"separator\":\"/\"}},"
  "\"fill_value\":0,"
  "\"codecs\":[{\"name\":\"sharding_indexed\",\"configuration\":{"
  "\"chunk_shape\":[2,4],"
  "\"codecs\":[{\"name\":\"bytes\",\"configuration\":{\"endian\":\"little\"}},"
  "{\"name\":\"zstd\",\"configuration\":{\"level\":3,\"checksum\":false}}],"
  "\"index_codecs\":[{\"name\":\"bytes\",\"configuration\":{"
  "\"endian\":\"little\"}},{\"name\":\"crc32c\"}],"
  "\"index_location\":\"end\"}}]"
  "}";

// Common fixture: tmpdir + foo array + one shard with 4 entries.
struct fixture
{
  char root[64];
  struct store* store;
  struct zarr_meta_cache* meta;
  struct zarr_shard_cache* shards;
  struct planner* planner;
};

static int
fixture_init(struct fixture* f, const uint64_t* offsets, const uint64_t* nbytes)
{
  strcpy(f->root, "/tmp/damacy_planner_XXXXXX");
  EXPECT(mkdtemp(f->root));
  char p[256];
  snprintf(p, sizeof p, "%s/foo", f->root);
  EXPECT(mkdir(p, 0755) == 0);
  snprintf(p, sizeof p, "%s/foo/zarr.json", f->root);
  EXPECT(fixture_write_file(p, MINIMAL_ZARR_JSON) == 0);
  snprintf(p, sizeof p, "%s/foo/c", f->root);
  EXPECT(mkdir(p, 0755) == 0);
  snprintf(p, sizeof p, "%s/foo/c/0", f->root);
  EXPECT(mkdir(p, 0755) == 0);
  snprintf(p, sizeof p, "%s/foo/c/0/0", f->root);
  // Payload bytes: large enough to hold any chunk we'll address. Last
  // offset+nbytes determines the lower bound; we use 4096 as headroom.
  EXPECT(fixture_write_synthetic_shard(p, 4096, offsets, nbytes, 4) == 0);

  struct store_fs_config sc = { .root = f->root, .nthreads = 1 };
  f->store = store_fs_create(&sc);
  EXPECT(f->store);
  f->meta = zarr_meta_cache_create(f->store, 4);
  EXPECT(f->meta);
  f->shards = zarr_shard_cache_create(f->store, 4);
  EXPECT(f->shards);
  struct planner_config pcfg = {
    .meta_cache = f->meta,
    .shard_cache = f->shards,
    .page_alignment = PAGE,
  };
  EXPECT(planner_create(&pcfg, &f->planner) == DAMACY_OK);
  return 0;
}

static void
fixture_destroy(struct fixture* f)
{
  planner_destroy(f->planner);
  zarr_shard_cache_destroy(f->shards);
  zarr_meta_cache_destroy(f->meta);
  store_destroy(f->store);
  fixture_rm_tree(f->root);
}

static struct damacy_sample
mk_sample(const char* uri, int64_t y0, int64_t y1, int64_t x0, int64_t x1)
{
  struct damacy_sample s = { .uri = uri, .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = y0, .end = y1 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = x0, .end = x1 };
  return s;
}

// Sample exactly covers one inner chunk. Expect 1 read_op + 1 chunk_plan.
static int
test_single_chunk_aligned(void)
{
  // 4 inner chunks (2 along y, 2 along x). Compressed sizes 32 each,
  // packed sequentially. Inner chunk uncompressed = 16 B. Compressed
  // > uncompressed for the worst case; we don't actually decompress.
  const uint64_t offsets[4] = { 0, 100, 200, 300 };
  const uint64_t nbytes[4] = { 32, 32, 32, 32 };
  struct fixture f = { 0 };
  if (fixture_init(&f, offsets, nbytes))
    return 1;

  // AABB = inner chunk (1, 0): y in [2,4), x in [0,4).
  struct damacy_sample s = mk_sample("foo", 2, 4, 0, 4);

  struct read_op reads[8] = { 0 };
  struct chunk_plan chunks[8] = { 0 };
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 8,
    .chunk_plans = chunks,
    .chunk_plans_cap = 8,
  };
  EXPECT(planner_plan(f.planner, &s, 1, 0, &out) == DAMACY_OK);

  EXPECT(out.n_chunk_plans == 1);
  EXPECT(out.n_read_ops == 1);

  // Inner chunk (1,0) is at shard-local linear 2 (row-major over [2,2]):
  // local_inner=(1,0) → 1*2 + 0 = 2 → offsets[2] = 200, nbytes 32.
  EXPECT(reads[0].file_offset == 0); // 200 page-aligned-down to 0
  EXPECT(reads[0].nbytes == PAGE);   // [200, 232) padded to [0, 4096)
  EXPECT(chunks[0].offset_in_read == 200);
  EXPECT(chunks[0].compressed_nbytes == 32);
  EXPECT(chunks[0].decompressed_nbytes == 16);
  EXPECT(chunks[0].batch_pool_slot == 0);
  EXPECT(chunks[0].read_op_idx == 0);

  // src AABB: full inner chunk (chunk-local).
  EXPECT(chunks[0].src.rank == 2);
  EXPECT(chunks[0].src.dims[0].beg == 0 && chunks[0].src.dims[0].end == 2);
  EXPECT(chunks[0].src.dims[1].beg == 0 && chunks[0].src.dims[1].end == 4);
  // dst AABB: [N=0..1, full sample]
  EXPECT(chunks[0].dst.rank == 3);
  EXPECT(chunks[0].dst.dims[0].beg == 0 && chunks[0].dst.dims[0].end == 1);
  EXPECT(chunks[0].dst.dims[1].beg == 0 && chunks[0].dst.dims[1].end == 2);
  EXPECT(chunks[0].dst.dims[2].beg == 0 && chunks[0].dst.dims[2].end == 4);
  // strides: row-major over [2, 4] → [4, 1]
  EXPECT(chunks[0].src_strides[0] == 4);
  EXPECT(chunks[0].src_strides[1] == 1);

  fixture_destroy(&f);
  return 0;
}

// Sample spans multiple inner chunks. Expect one chunk_plan per chunk
// touched, with the right intersection windows.
static int
test_multi_chunk_partial(void)
{
  const uint64_t offsets[4] = { 0, 100, 200, 300 };
  const uint64_t nbytes[4] = { 32, 32, 32, 32 };
  struct fixture f = { 0 };
  if (fixture_init(&f, offsets, nbytes))
    return 1;

  // AABB = y in [1,4), x in [2,7). Touches all 4 chunks (2x2 grid).
  // Chunk (0,0) covers y[0,2) x[0,4): intersection y[1,2) x[2,4)
  // Chunk (0,1) covers y[0,2) x[4,8): intersection y[1,2) x[4,7)
  // Chunk (1,0) covers y[2,4) x[0,4): intersection y[2,4) x[2,4)
  // Chunk (1,1) covers y[2,4) x[4,8): intersection y[2,4) x[4,7)
  struct damacy_sample s = mk_sample("foo", 1, 4, 2, 7);

  struct read_op reads[8] = { 0 };
  struct chunk_plan chunks[8] = { 0 };
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 8,
    .chunk_plans = chunks,
    .chunk_plans_cap = 8,
  };
  EXPECT(planner_plan(f.planner, &s, 1, 0, &out) == DAMACY_OK);
  EXPECT(out.n_chunk_plans == 4);
  EXPECT(out.n_read_ops == 4);

  // Chunks emitted in row-major order: (0,0), (0,1), (1,0), (1,1).
  // src dims (chunk-local):
  EXPECT(chunks[0].src.dims[0].beg == 1 && chunks[0].src.dims[0].end == 2);
  EXPECT(chunks[0].src.dims[1].beg == 2 && chunks[0].src.dims[1].end == 4);
  EXPECT(chunks[1].src.dims[0].beg == 1 && chunks[1].src.dims[0].end == 2);
  EXPECT(chunks[1].src.dims[1].beg == 0 && chunks[1].src.dims[1].end == 3);
  EXPECT(chunks[2].src.dims[0].beg == 0 && chunks[2].src.dims[0].end == 2);
  EXPECT(chunks[2].src.dims[1].beg == 2 && chunks[2].src.dims[1].end == 4);
  EXPECT(chunks[3].src.dims[0].beg == 0 && chunks[3].src.dims[0].end == 2);
  EXPECT(chunks[3].src.dims[1].beg == 0 && chunks[3].src.dims[1].end == 3);
  // dst dims (sample-local, with leading N=0):
  // sample y[1,4) → local y[0,3); sample x[2,7) → local x[0,5)
  EXPECT(chunks[0].dst.dims[1].beg == 0 && chunks[0].dst.dims[1].end == 1);
  EXPECT(chunks[0].dst.dims[2].beg == 0 && chunks[0].dst.dims[2].end == 2);
  EXPECT(chunks[3].dst.dims[1].beg == 1 && chunks[3].dst.dims[1].end == 3);
  EXPECT(chunks[3].dst.dims[2].beg == 2 && chunks[3].dst.dims[2].end == 5);
  // All chunks share batch_pool_slot 0 and N=(0,1).
  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT(chunks[i].batch_pool_slot == 0);
    EXPECT(chunks[i].dst.dims[0].beg == 0 && chunks[i].dst.dims[0].end == 1);
  }

  fixture_destroy(&f);
  return 0;
}

// Two samples in one batch: dst.dims[0] differs.
static int
test_two_samples_indices(void)
{
  const uint64_t offsets[4] = { 0, 100, 200, 300 };
  const uint64_t nbytes[4] = { 32, 32, 32, 32 };
  struct fixture f = { 0 };
  if (fixture_init(&f, offsets, nbytes))
    return 1;

  struct damacy_sample s[2] = {
    mk_sample("foo", 0, 2, 0, 4), // chunk (0,0)
    mk_sample("foo", 2, 4, 4, 8), // chunk (1,1)
  };

  struct read_op reads[8] = { 0 };
  struct chunk_plan chunks[8] = { 0 };
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 8,
    .chunk_plans = chunks,
    .chunk_plans_cap = 8,
  };
  EXPECT(planner_plan(f.planner, s, 2, 7, &out) == DAMACY_OK);
  EXPECT(out.n_chunk_plans == 2);
  EXPECT(chunks[0].dst.dims[0].beg == 0 && chunks[0].dst.dims[0].end == 1);
  EXPECT(chunks[1].dst.dims[0].beg == 1 && chunks[1].dst.dims[0].end == 2);
  EXPECT(chunks[0].batch_pool_slot == 7);
  EXPECT(chunks[1].batch_pool_slot == 7);

  fixture_destroy(&f);
  return 0;
}

// Empty chunks (offset == sentinel) are skipped.
static int
test_empty_chunks_skipped(void)
{
  // Mark chunk (1,0) as empty (linear index 2).
  const uint64_t offsets[4] = { 0, 100, ZARR_SHARD_EMPTY_OFFSET, 300 };
  const uint64_t nbytes[4] = { 32, 32, ZARR_SHARD_EMPTY_NBYTES, 32 };
  struct fixture f = { 0 };
  if (fixture_init(&f, offsets, nbytes))
    return 1;

  struct damacy_sample s = mk_sample("foo", 0, 4, 0, 8); // all 4 chunks

  struct read_op reads[8] = { 0 };
  struct chunk_plan chunks[8] = { 0 };
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 8,
    .chunk_plans = chunks,
    .chunk_plans_cap = 8,
  };
  EXPECT(planner_plan(f.planner, &s, 1, 0, &out) == DAMACY_OK);
  EXPECT(out.n_chunk_plans == 3); // (1,0) skipped
  EXPECT(out.n_read_ops == 3);

  fixture_destroy(&f);
  return 0;
}

// Non-page-aligned offsets: read_op should round down to a page boundary
// and round up to a page-multiple length.
static int
test_page_alignment(void)
{
  // Pick offsets that straddle page boundaries.
  const uint64_t offsets[4] = { 4090, 4096 + 50, 8192 + 200, 12288 + 4090 };
  const uint64_t nbytes[4] = { 32, 32, 32, 32 };
  struct fixture f = { 0 };
  if (fixture_init(&f, offsets, nbytes))
    return 1;

  struct damacy_sample s = mk_sample("foo", 0, 4, 0, 8);
  struct read_op reads[8] = { 0 };
  struct chunk_plan chunks[8] = { 0 };
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 8,
    .chunk_plans = chunks,
    .chunk_plans_cap = 8,
  };
  EXPECT(planner_plan(f.planner, &s, 1, 0, &out) == DAMACY_OK);

  for (uint32_t i = 0; i < out.n_read_ops; ++i) {
    EXPECT(reads[i].file_offset % PAGE == 0);
    EXPECT(reads[i].nbytes % PAGE == 0);
  }

  // Chunk (0,0): offset 4090 → file_offset 0, nbytes 8192 (covers [4090,4122))
  // chunk_offset_in_read = 4090
  EXPECT(reads[0].file_offset == 0);
  EXPECT(reads[0].nbytes == 2 * PAGE);
  EXPECT(chunks[0].offset_in_read == 4090);

  // Chunk (0,1): offset 4146 → file_offset 4096, nbytes 4096
  EXPECT(reads[1].file_offset == 4096);
  EXPECT(reads[1].nbytes == PAGE);
  EXPECT(chunks[1].offset_in_read == 50);

  fixture_destroy(&f);
  return 0;
}

int
main(void)
{
  RUN(test_single_chunk_aligned);
  RUN(test_multi_chunk_partial);
  RUN(test_two_samples_indices);
  RUN(test_empty_chunks_skipped);
  RUN(test_page_alignment);
  log_info("all tests passed");
  return 0;
}
