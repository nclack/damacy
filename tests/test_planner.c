// Integration test for the planner. Spins up a tmpdir-backed fs store
// with a synthetic 2D zarr (one shard, four inner chunks) and verifies
// planner output: read counts, page alignment, sample_plan dims,
// chunk grid coordinates.

#include "damacy.h"
#include "fixture.h"
#include "planner/planner.h"
#include "store/store.h"
#include "zarr/zarr_meta_cache.h"
#include "zarr/zarr_metadata.h"
#include "zarr/zarr_shard_cache.h"
#include "zarr/zarr_shard_index.h"

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

// Same geometry but fill_value=-1 with int16 dtype, no zstd inner codec
// (raw bytes) — keeps the fill_value bytes path simple for inspection.
static const char* INT16_FILL_NEG1_ZARR_JSON =
  "{"
  "\"zarr_format\":3,"
  "\"node_type\":\"array\","
  "\"shape\":[4,8],"
  "\"data_type\":\"int16\","
  "\"chunk_grid\":{\"name\":\"regular\",\"configuration\":{"
  "\"chunk_shape\":[4,8]}},"
  "\"chunk_key_encoding\":{\"name\":\"default\",\"configuration\":{"
  "\"separator\":\"/\"}},"
  "\"fill_value\":-1,"
  "\"codecs\":[{\"name\":\"sharding_indexed\",\"configuration\":{"
  "\"chunk_shape\":[2,4],"
  "\"codecs\":[{\"name\":\"bytes\",\"configuration\":{\"endian\":\"little\"}}],"
  "\"index_codecs\":[{\"name\":\"bytes\",\"configuration\":{"
  "\"endian\":\"little\"}},{\"name\":\"crc32c\"}],"
  "\"index_location\":\"end\"}}]"
  "}";

// float32 fill_value="NaN" (string-form per zarr v3).
static const char* F32_FILL_NAN_ZARR_JSON =
  "{"
  "\"zarr_format\":3,"
  "\"node_type\":\"array\","
  "\"shape\":[4,8],"
  "\"data_type\":\"float32\","
  "\"chunk_grid\":{\"name\":\"regular\",\"configuration\":{"
  "\"chunk_shape\":[4,8]}},"
  "\"chunk_key_encoding\":{\"name\":\"default\",\"configuration\":{"
  "\"separator\":\"/\"}},"
  "\"fill_value\":\"NaN\","
  "\"codecs\":[{\"name\":\"sharding_indexed\",\"configuration\":{"
  "\"chunk_shape\":[2,4],"
  "\"codecs\":[{\"name\":\"bytes\",\"configuration\":{\"endian\":\"little\"}}],"
  "\"index_codecs\":[{\"name\":\"bytes\",\"configuration\":{"
  "\"endian\":\"little\"}},{\"name\":\"crc32c\"}],"
  "\"index_location\":\"end\"}}]"
  "}";

// Same as MINIMAL_ZARR_JSON but inner codec list contains only "bytes"
// (no compressor) — the path that yields CODEC_NONE.
static const char* NONE_ZARR_JSON =
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
  "\"codecs\":[{\"name\":\"bytes\",\"configuration\":{\"endian\":\"little\"}}],"
  "\"index_codecs\":[{\"name\":\"bytes\",\"configuration\":{"
  "\"endian\":\"little\"}},{\"name\":\"crc32c\"}],"
  "\"index_location\":\"end\"}}]"
  "}";

// Same as MINIMAL_ZARR_JSON but inner codec is "blosc"; %s is the cname.
static const char* BLOSC_ZARR_JSON_FMT =
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
  "{\"name\":\"blosc\",\"configuration\":{"
  "\"cname\":\"%s\",\"clevel\":3,\"shuffle\":\"shuffle\","
  "\"typesize\":2,\"blocksize\":0}}],"
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
fixture_init_with_json(struct fixture* f,
                       const char* zarr_json,
                       const uint64_t* offsets,
                       const uint64_t* nbytes)
{
  strcpy(f->root, "/tmp/damacy_planner_XXXXXX");
  EXPECT(mkdtemp(f->root));
  char p[256];
  snprintf(p, sizeof p, "%s/foo", f->root);
  EXPECT(mkdir(p, 0755) == 0);
  snprintf(p, sizeof p, "%s/foo/zarr.json", f->root);
  EXPECT(fixture_write_file(p, zarr_json) == 0);
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

static int
fixture_init(struct fixture* f, const uint64_t* offsets, const uint64_t* nbytes)
{
  return fixture_init_with_json(f, MINIMAL_ZARR_JSON, offsets, nbytes);
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

// Row-major dst strides for output shape [N, H, W].
static void
mk_dst_strides_2d(int64_t n, int64_t h, int64_t w, int64_t* out)
{
  (void)n;
  out[2] = 1;
  out[1] = w;
  out[0] = h * w;
}

// Sample exactly covers one inner chunk. Expect 1 read_op + 1 chunk_plan
// + 1 sample_plan with N=[1,1], chunk_d=(0,0).
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
  int64_t dst_strides[3];
  mk_dst_strides_2d(1, 2, 4, dst_strides);

  struct read_op reads[8] = { 0 };
  struct chunk_plan chunks[8] = { 0 };
  struct sample_plan samples[4] = { 0 };
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 8,
    .chunk_plans = chunks,
    .chunk_plans_cap = 8,
    .sample_plans = samples,
    .sample_plans_cap = 4,
  };
  EXPECT(planner_plan(f.planner, &s, 1, 0, dst_strides, 3, &out) == DAMACY_OK);

  EXPECT(out.n_chunk_plans == 1);
  EXPECT(out.n_read_ops == 1);
  EXPECT(out.n_sample_plans == 1);

  // Inner chunk (1,0) is at shard-local linear 2 (row-major over [2,2]):
  // local_inner=(1,0) → 1*2 + 0 = 2 → offsets[2] = 200, nbytes 32.
  EXPECT(reads[0].file_offset == 0); // 200 page-aligned-down to 0
  EXPECT(reads[0].nbytes == PAGE);   // [200, 232) padded to [0, 4096)
  EXPECT(chunks[0].offset_in_read == 200);
  EXPECT(chunks[0].compressed_nbytes == 32);
  EXPECT(chunks[0].decompressed_nbytes == 16);
  EXPECT(chunks[0].batch_pool_slot == 0);
  EXPECT(chunks[0].read_op_idx == 0);
  EXPECT(chunks[0].sample_idx_in_batch == 0);
  EXPECT(chunks[0].chunk_d[0] == 0 && chunks[0].chunk_d[1] == 0);
  EXPECT(chunks[0].codec_id == CODEC_ZSTD);

  // Sample plan: rank 2, N=[1,1] (one chunk), S=[2,4], aabb_extent=[2,4],
  // aabb_lo_relative=[0,0] (sample starts at chunk grid origin).
  struct sample_plan* sp = &samples[0];
  EXPECT(sp->rank == 2);
  EXPECT(sp->batch_pool_slot == 0);
  EXPECT(sp->sample_idx_in_batch == 0);
  EXPECT(sp->chunk_offset == 0);
  EXPECT(sp->chunk_count == 1);
  EXPECT(sp->dims[0].chunk_shape == 2 && sp->dims[1].chunk_shape == 4);
  EXPECT(sp->dims[0].chunk_grid_extent == 1 &&
         sp->dims[1].chunk_grid_extent == 1);
  EXPECT(sp->dims[0].aabb_lo_relative == 0 &&
         sp->dims[1].aabb_lo_relative == 0);
  EXPECT(sp->dims[0].aabb_extent == 2 && sp->dims[1].aabb_extent == 4);
  // src strides: row-major over [2, 4] → [4, 1]
  EXPECT(sp->dims[0].src_stride == 4 && sp->dims[1].src_stride == 1);
  // dst strides: [W, 1] = [4, 1]
  EXPECT(sp->dims[0].dst_stride == 4 && sp->dims[1].dst_stride == 1);
  EXPECT(sp->sample_dst_off_elems == 0);

  fixture_destroy(&f);
  return 0;
}

// Sample spans multiple inner chunks. Expect one chunk_plan per chunk
// with row-major chunk_d, and a single sample_plan with N=[2,2] and
// aabb_lo_relative reflecting where the AABB starts within chunk (0,0).
static int
test_multi_chunk_partial(void)
{
  const uint64_t offsets[4] = { 0, 100, 200, 300 };
  const uint64_t nbytes[4] = { 32, 32, 32, 32 };
  struct fixture f = { 0 };
  if (fixture_init(&f, offsets, nbytes))
    return 1;

  // AABB = y in [1,4), x in [2,7). Touches all 4 chunks (2x2 grid).
  // Chunk grid origin: (0,0). aabb_lo=(1,2). aabb_lo_relative=(1,2).
  struct damacy_sample s = mk_sample("foo", 1, 4, 2, 7);
  int64_t dst_strides[3];
  mk_dst_strides_2d(1, 3, 5, dst_strides);

  struct read_op reads[8] = { 0 };
  struct chunk_plan chunks[8] = { 0 };
  struct sample_plan samples[4] = { 0 };
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 8,
    .chunk_plans = chunks,
    .chunk_plans_cap = 8,
    .sample_plans = samples,
    .sample_plans_cap = 4,
  };
  EXPECT(planner_plan(f.planner, &s, 1, 0, dst_strides, 3, &out) == DAMACY_OK);
  EXPECT(out.n_chunk_plans == 4);
  EXPECT(out.n_read_ops == 4);
  EXPECT(out.n_sample_plans == 1);

  // Chunks emitted in row-major chunk_d order: (0,0), (0,1), (1,0), (1,1).
  EXPECT(chunks[0].chunk_d[0] == 0 && chunks[0].chunk_d[1] == 0);
  EXPECT(chunks[1].chunk_d[0] == 0 && chunks[1].chunk_d[1] == 1);
  EXPECT(chunks[2].chunk_d[0] == 1 && chunks[2].chunk_d[1] == 0);
  EXPECT(chunks[3].chunk_d[0] == 1 && chunks[3].chunk_d[1] == 1);
  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT(chunks[i].batch_pool_slot == 0);
    EXPECT(chunks[i].sample_idx_in_batch == 0);
  }

  struct sample_plan* sp = &samples[0];
  EXPECT(sp->rank == 2);
  EXPECT(sp->chunk_offset == 0);
  EXPECT(sp->chunk_count == 4);
  EXPECT(sp->dims[0].chunk_grid_extent == 2 &&
         sp->dims[1].chunk_grid_extent == 2);
  EXPECT(sp->dims[0].aabb_lo_relative == 1 &&
         sp->dims[1].aabb_lo_relative == 2);
  EXPECT(sp->dims[0].aabb_extent == 3 && sp->dims[1].aabb_extent == 5);

  fixture_destroy(&f);
  return 0;
}

// Two samples in one batch: sample_idx_in_batch differs; sample_dst_off
// matches sample_idx * dst_strides[0].
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
  int64_t dst_strides[3];
  mk_dst_strides_2d(2, 2, 4, dst_strides); // strides=[8,4,1]

  struct read_op reads[8] = { 0 };
  struct chunk_plan chunks[8] = { 0 };
  struct sample_plan samples[4] = { 0 };
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 8,
    .chunk_plans = chunks,
    .chunk_plans_cap = 8,
    .sample_plans = samples,
    .sample_plans_cap = 4,
  };
  EXPECT(planner_plan(f.planner, s, 2, 7, dst_strides, 3, &out) == DAMACY_OK);
  EXPECT(out.n_chunk_plans == 2);
  EXPECT(out.n_sample_plans == 2);

  EXPECT(chunks[0].sample_idx_in_batch == 0);
  EXPECT(chunks[1].sample_idx_in_batch == 1);
  EXPECT(chunks[0].batch_pool_slot == 7);
  EXPECT(chunks[1].batch_pool_slot == 7);

  EXPECT(samples[0].sample_dst_off_elems == 0);
  EXPECT(samples[1].sample_dst_off_elems == 8); // sample_idx=1 * stride[0]=8
  EXPECT(samples[0].chunk_offset == 0);
  EXPECT(samples[1].chunk_offset == 1);
  EXPECT(samples[0].chunk_count == 1 && samples[1].chunk_count == 1);

  fixture_destroy(&f);
  return 0;
}

// Empty inner chunks inside a sample's AABB are emitted as fill-mode
// chunk plans referencing the array's fill_value.
static int
test_empty_chunk_becomes_fill(void)
{
  // Mark chunk (1,0) as empty (linear index 2).
  const uint64_t offsets[4] = { 0, 100, ZARR_SHARD_EMPTY_OFFSET, 300 };
  const uint64_t nbytes[4] = { 32, 32, ZARR_SHARD_EMPTY_NBYTES, 32 };
  struct fixture f = { 0 };
  if (fixture_init(&f, offsets, nbytes))
    return 1;

  struct damacy_sample s = mk_sample("foo", 0, 4, 0, 8); // all 4 chunks
  int64_t dst_strides[3];
  mk_dst_strides_2d(1, 4, 8, dst_strides);

  struct read_op reads[8] = { 0 };
  struct chunk_plan chunks[8] = { 0 };
  struct sample_plan samples[4] = { 0 };
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 8,
    .chunk_plans = chunks,
    .chunk_plans_cap = 8,
    .sample_plans = samples,
    .sample_plans_cap = 4,
  };
  EXPECT(planner_plan(f.planner, &s, 1, 0, dst_strides, 3, &out) == DAMACY_OK);
  EXPECT(out.n_chunk_plans == 4);
  // Row-major (0,0), (0,1), (1,0)*, (1,1) — the third is fill.
  EXPECT(chunks[0].is_fill == 0);
  EXPECT(chunks[1].is_fill == 0);
  EXPECT(chunks[2].is_fill == 1);
  EXPECT(chunks[3].is_fill == 0);
  EXPECT(chunks[2].codec_id == CODEC_FILL);
  EXPECT(chunks[2].compressed_nbytes == 0);
  // fill_value parsed from "fill_value":0 in the JSON → all zero bytes.
  for (size_t k = 0; k < sizeof chunks[2].fill_value; ++k)
    EXPECT(chunks[2].fill_value[k] == 0);

  fixture_destroy(&f);
  return 0;
}

// int16 fill_value=-1 → little-endian bytes 0xFF 0xFF on the chunk plan
// when the corresponding inner chunk is empty.
static int
test_fill_value_int16_neg1(void)
{
  const uint64_t offsets[4] = { 0, ZARR_SHARD_EMPTY_OFFSET, 200, 300 };
  const uint64_t nbytes[4] = { 32, ZARR_SHARD_EMPTY_NBYTES, 32, 32 };
  struct fixture f = { 0 };
  if (fixture_init_with_json(&f, INT16_FILL_NEG1_ZARR_JSON, offsets, nbytes))
    return 1;

  struct damacy_sample s = mk_sample("foo", 0, 4, 0, 8);
  int64_t dst_strides[3];
  mk_dst_strides_2d(1, 4, 8, dst_strides);

  struct read_op reads[8] = { 0 };
  struct chunk_plan chunks[8] = { 0 };
  struct sample_plan samples[4] = { 0 };
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 8,
    .chunk_plans = chunks,
    .chunk_plans_cap = 8,
    .sample_plans = samples,
    .sample_plans_cap = 4,
  };
  EXPECT(planner_plan(f.planner, &s, 1, 0, dst_strides, 3, &out) == DAMACY_OK);
  EXPECT(out.n_chunk_plans == 4);
  // chunk index 1 = (0,1) is empty → fill.
  EXPECT(chunks[1].is_fill == 1);
  EXPECT(chunks[1].fill_value[0] == 0xFF);
  EXPECT(chunks[1].fill_value[1] == 0xFF);
  fixture_destroy(&f);
  return 0;
}

// float32 fill_value="NaN" parses into a NaN bit pattern.
static int
test_fill_value_f32_nan(void)
{
  const uint64_t offsets[4] = { 0, ZARR_SHARD_EMPTY_OFFSET, 200, 300 };
  const uint64_t nbytes[4] = { 32, ZARR_SHARD_EMPTY_NBYTES, 32, 32 };
  struct fixture f = { 0 };
  if (fixture_init_with_json(&f, F32_FILL_NAN_ZARR_JSON, offsets, nbytes))
    return 1;

  struct damacy_sample s = mk_sample("foo", 0, 4, 0, 8);
  int64_t dst_strides[3];
  mk_dst_strides_2d(1, 4, 8, dst_strides);
  struct read_op reads[8] = { 0 };
  struct chunk_plan chunks[8] = { 0 };
  struct sample_plan samples[4] = { 0 };
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 8,
    .chunk_plans = chunks,
    .chunk_plans_cap = 8,
    .sample_plans = samples,
    .sample_plans_cap = 4,
  };
  EXPECT(planner_plan(f.planner, &s, 1, 0, dst_strides, 3, &out) == DAMACY_OK);
  EXPECT(chunks[1].is_fill == 1);
  float fill_f;
  memcpy(&fill_f, chunks[1].fill_value, sizeof fill_f);
  EXPECT(fill_f != fill_f); // NaN is the unique value that compares unequal
  fixture_destroy(&f);
  return 0;
}

// Sparse shard file (zarr_shard_cache_get → DAMACY_NOTFOUND) makes the
// whole shard's chunks fill-mode.
static int
test_missing_shard_becomes_fill(void)
{
  // Same JSON, but no shard file at all under foo/c/0/0.
  char root[] = "/tmp/damacy_planner_XXXXXX";
  EXPECT(mkdtemp(root));
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);
  EXPECT(mkdir(p, 0755) == 0);
  snprintf(p, sizeof p, "%s/foo/zarr.json", root);
  EXPECT(fixture_write_file(p, MINIMAL_ZARR_JSON) == 0);
  // Intentionally do NOT create foo/c/0/0.

  struct store_fs_config sc = { .root = root, .nthreads = 1 };
  struct store* store = store_fs_create(&sc);
  EXPECT(store);
  struct zarr_meta_cache* meta = zarr_meta_cache_create(store, 4);
  EXPECT(meta);
  struct zarr_shard_cache* shards = zarr_shard_cache_create(store, 4);
  EXPECT(shards);
  struct planner_config pcfg = {
    .meta_cache = meta,
    .shard_cache = shards,
    .page_alignment = PAGE,
  };
  struct planner* planner = NULL;
  EXPECT(planner_create(&pcfg, &planner) == DAMACY_OK);

  struct damacy_sample s = mk_sample("foo", 0, 4, 0, 8);
  int64_t dst_strides[3];
  mk_dst_strides_2d(1, 4, 8, dst_strides);
  struct read_op reads[8] = { 0 };
  struct chunk_plan chunks[8] = { 0 };
  struct sample_plan samples[4] = { 0 };
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 8,
    .chunk_plans = chunks,
    .chunk_plans_cap = 8,
    .sample_plans = samples,
    .sample_plans_cap = 4,
  };
  EXPECT(planner_plan(planner, &s, 1, 0, dst_strides, 3, &out) == DAMACY_OK);
  EXPECT(out.n_chunk_plans == 4);
  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT(chunks[i].is_fill == 1);
    EXPECT(chunks[i].codec_id == CODEC_FILL);
  }

  planner_destroy(planner);
  zarr_shard_cache_destroy(shards);
  zarr_meta_cache_destroy(meta);
  store_destroy(store);
  fixture_rm_tree(root);
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
  int64_t dst_strides[3];
  mk_dst_strides_2d(1, 4, 8, dst_strides);
  struct read_op reads[8] = { 0 };
  struct chunk_plan chunks[8] = { 0 };
  struct sample_plan samples[4] = { 0 };
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 8,
    .chunk_plans = chunks,
    .chunk_plans_cap = 8,
    .sample_plans = samples,
    .sample_plans_cap = 4,
  };
  EXPECT(planner_plan(f.planner, &s, 1, 0, dst_strides, 3, &out) == DAMACY_OK);

  for (uint32_t i = 0; i < out.n_read_ops; ++i) {
    EXPECT(reads[i].file_offset % PAGE == 0);
    EXPECT(reads[i].nbytes % PAGE == 0);
  }

  // Chunk (0,0): offset 4090 → file_offset 0, nbytes 8192 (covers [4090,4122))
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

static int
run_blosc_codec_id_case(const char* cname, uint8_t expected_codec_id)
{
  char json[1024];
  int n = snprintf(json, sizeof json, BLOSC_ZARR_JSON_FMT, cname);
  EXPECT(n > 0 && (size_t)n < sizeof json);

  const uint64_t offsets[4] = { 0, 100, 200, 300 };
  const uint64_t nbytes[4] = { 32, 32, 32, 32 };
  struct fixture f = { 0 };
  if (fixture_init_with_json(&f, json, offsets, nbytes))
    return 1;

  struct damacy_sample s = mk_sample("foo", 0, 4, 0, 8);
  int64_t dst_strides[3];
  mk_dst_strides_2d(1, 4, 8, dst_strides);

  struct read_op reads[8] = { 0 };
  struct chunk_plan chunks[8] = { 0 };
  struct sample_plan samples[4] = { 0 };
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 8,
    .chunk_plans = chunks,
    .chunk_plans_cap = 8,
    .sample_plans = samples,
    .sample_plans_cap = 4,
  };
  EXPECT(planner_plan(f.planner, &s, 1, 0, dst_strides, 3, &out) == DAMACY_OK);
  EXPECT(out.n_chunk_plans == 4);
  for (uint32_t i = 0; i < out.n_chunk_plans; ++i)
    EXPECT(chunks[i].codec_id == expected_codec_id);

  fixture_destroy(&f);
  return 0;
}

// Planner rejects blosc1-lz4 with DAMACY_INVAL (codec is no longer supported).
static int
run_blosc_lz4_rejected_case(const char* cname)
{
  char json[1024];
  int n = snprintf(json, sizeof json, BLOSC_ZARR_JSON_FMT, cname);
  EXPECT(n > 0 && (size_t)n < sizeof json);

  const uint64_t offsets[4] = { 0, 100, 200, 300 };
  const uint64_t nbytes[4] = { 32, 32, 32, 32 };
  struct fixture f = { 0 };
  if (fixture_init_with_json(&f, json, offsets, nbytes))
    return 1;

  struct damacy_sample s = mk_sample("foo", 0, 4, 0, 8);
  int64_t dst_strides[3];
  mk_dst_strides_2d(1, 4, 8, dst_strides);

  struct read_op reads[8] = { 0 };
  struct chunk_plan chunks[8] = { 0 };
  struct sample_plan samples[4] = { 0 };
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 8,
    .chunk_plans = chunks,
    .chunk_plans_cap = 8,
    .sample_plans = samples,
    .sample_plans_cap = 4,
  };
  EXPECT(planner_plan(f.planner, &s, 1, 0, dst_strides, 3, &out) ==
         DAMACY_INVAL);

  fixture_destroy(&f);
  return 0;
}

static int
test_codec_id_blosc_lz4_rejected(void)
{
  return run_blosc_lz4_rejected_case("lz4");
}

static int
test_codec_id_blosc_lz4hc_rejected(void)
{
  return run_blosc_lz4_rejected_case("lz4hc");
}

static int
test_codec_id_blosc_zstd(void)
{
  return run_blosc_codec_id_case("zstd", CODEC_BLOSC_ZSTD);
}

static int
test_codec_id_none(void)
{
  const uint64_t offsets[4] = { 0, 100, 200, 300 };
  const uint64_t nbytes[4] = { 32, 32, 32, 32 };
  struct fixture f = { 0 };
  if (fixture_init_with_json(&f, NONE_ZARR_JSON, offsets, nbytes))
    return 1;

  struct damacy_sample s = mk_sample("foo", 0, 4, 0, 8);
  int64_t dst_strides[3];
  mk_dst_strides_2d(1, 4, 8, dst_strides);

  struct read_op reads[8] = { 0 };
  struct chunk_plan chunks[8] = { 0 };
  struct sample_plan samples[4] = { 0 };
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 8,
    .chunk_plans = chunks,
    .chunk_plans_cap = 8,
    .sample_plans = samples,
    .sample_plans_cap = 4,
  };
  EXPECT(planner_plan(f.planner, &s, 1, 0, dst_strides, 3, &out) == DAMACY_OK);
  EXPECT(out.n_chunk_plans == 4);
  for (uint32_t i = 0; i < out.n_chunk_plans; ++i)
    EXPECT(chunks[i].codec_id == CODEC_NONE);

  fixture_destroy(&f);
  return 0;
}

// Unknown blosc cname must propagate as a non-OK status; the resolver
// only accepts {lz4, lz4hc, zstd}.
static int
test_codec_id_blosc_unknown_cname(void)
{
  char json[1024];
  int n = snprintf(json, sizeof json, BLOSC_ZARR_JSON_FMT, "bzip2");
  EXPECT(n > 0 && (size_t)n < sizeof json);

  const uint64_t offsets[4] = { 0, 100, 200, 300 };
  const uint64_t nbytes[4] = { 32, 32, 32, 32 };
  struct fixture f = { 0 };
  if (fixture_init_with_json(&f, json, offsets, nbytes))
    return 1;

  struct damacy_sample s = mk_sample("foo", 0, 4, 0, 8);
  int64_t dst_strides[3];
  mk_dst_strides_2d(1, 4, 8, dst_strides);

  struct read_op reads[8] = { 0 };
  struct chunk_plan chunks[8] = { 0 };
  struct sample_plan samples[4] = { 0 };
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 8,
    .chunk_plans = chunks,
    .chunk_plans_cap = 8,
    .sample_plans = samples,
    .sample_plans_cap = 4,
  };
  EXPECT(planner_plan(f.planner, &s, 1, 0, dst_strides, 3, &out) != DAMACY_OK);

  fixture_destroy(&f);
  return 0;
}

// Non-sharded zarr v3: each chunk is its own file at c/<i>/<j>/...
// inner_chunk_shape == chunk_grid's chunk_shape; the cache synthesises
// a 1-entry index per file. Verify the planner plans correctly.
static const char* UNSHARDED_ZSTD_ZARR_JSON =
  "{"
  "\"zarr_format\":3,"
  "\"node_type\":\"array\","
  "\"shape\":[4,8],"
  "\"data_type\":\"uint16\","
  "\"chunk_grid\":{\"name\":\"regular\",\"configuration\":{"
  "\"chunk_shape\":[2,4]}},"
  "\"chunk_key_encoding\":{\"name\":\"default\",\"configuration\":{"
  "\"separator\":\"/\"}},"
  "\"fill_value\":0,"
  "\"codecs\":[{\"name\":\"bytes\",\"configuration\":{\"endian\":\"little\"}},"
  "{\"name\":\"zstd\",\"configuration\":{\"level\":3}}]"
  "}";

// Sharded zarr v3 with index_location: "start" (otherwise identical to
// MINIMAL_ZARR_JSON).
static const char* SHARDED_INDEX_START_ZARR_JSON =
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
  "\"index_location\":\"start\"}}]"
  "}";

// Build a fixture for a non-sharded array. Each chunk c/<i>/<j> is a
// plain file whose entire content is the (synthetic) compressed chunk.
static int
fixture_init_unsharded(struct fixture* f, const uint32_t* chunk_bytes_2x2)
{
  strcpy(f->root, "/tmp/damacy_planner_un_XXXXXX");
  EXPECT(mkdtemp(f->root));
  char p[256];
  snprintf(p, sizeof p, "%s/foo", f->root);
  EXPECT(mkdir(p, 0755) == 0);
  snprintf(p, sizeof p, "%s/foo/zarr.json", f->root);
  EXPECT(fixture_write_file(p, UNSHARDED_ZSTD_ZARR_JSON) == 0);
  snprintf(p, sizeof p, "%s/foo/c", f->root);
  EXPECT(mkdir(p, 0755) == 0);
  for (uint8_t i = 0; i < 2; ++i) {
    snprintf(p, sizeof p, "%s/foo/c/%u", f->root, i);
    EXPECT(mkdir(p, 0755) == 0);
    for (uint8_t j = 0; j < 2; ++j) {
      snprintf(p, sizeof p, "%s/foo/c/%u/%u", f->root, i, j);
      EXPECT(fixture_write_zero_file(p, chunk_bytes_2x2[i * 2 + j]) == 0);
    }
  }

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

static int
test_unsharded_single_chunk(void)
{
  // 2x2 grid of chunks; each file is its own "shard of one".
  const uint32_t chunk_bytes[4] = { 32, 40, 48, 56 };
  struct fixture f = { 0 };
  if (fixture_init_unsharded(&f, chunk_bytes))
    return 1;

  // Sample covers exactly chunk (1, 0): y in [2,4), x in [0,4).
  struct damacy_sample s = mk_sample("foo", 2, 4, 0, 4);
  int64_t dst_strides[3];
  mk_dst_strides_2d(1, 2, 4, dst_strides);

  struct read_op reads[8] = { 0 };
  struct chunk_plan chunks[8] = { 0 };
  struct sample_plan samples[4] = { 0 };
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 8,
    .chunk_plans = chunks,
    .chunk_plans_cap = 8,
    .sample_plans = samples,
    .sample_plans_cap = 4,
  };
  EXPECT(planner_plan(f.planner, &s, 1, 0, dst_strides, 3, &out) == DAMACY_OK);

  EXPECT(out.n_chunk_plans == 1);
  EXPECT(out.n_read_ops == 1);
  EXPECT(out.n_sample_plans == 1);

  // The "chunk file" is c/1/0; offset 0; nbytes 48 (chunk_bytes[2]).
  // shard_path constructed inside the planner is "foo/c/1/0".
  EXPECT(strcmp(reads[0].shard_path, "foo/c/1/0") == 0);
  EXPECT(reads[0].file_offset == 0);
  EXPECT(reads[0].nbytes == PAGE);
  EXPECT(chunks[0].offset_in_read == 0);
  EXPECT(chunks[0].compressed_nbytes == 48);
  EXPECT(chunks[0].decompressed_nbytes == 16);
  EXPECT(chunks[0].codec_id == CODEC_ZSTD);
  EXPECT(chunks[0].chunk_d[0] == 0 && chunks[0].chunk_d[1] == 0);

  fixture_destroy(&f);
  return 0;
}

static int
test_unsharded_multi_chunk(void)
{
  // Sample spans all 4 chunks; one read_op per chunk file.
  const uint32_t chunk_bytes[4] = { 30, 40, 50, 60 };
  struct fixture f = { 0 };
  if (fixture_init_unsharded(&f, chunk_bytes))
    return 1;

  struct damacy_sample s = mk_sample("foo", 0, 4, 0, 8);
  int64_t dst_strides[3];
  mk_dst_strides_2d(1, 4, 8, dst_strides);

  struct read_op reads[8] = { 0 };
  struct chunk_plan chunks[8] = { 0 };
  struct sample_plan samples[4] = { 0 };
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 8,
    .chunk_plans = chunks,
    .chunk_plans_cap = 8,
    .sample_plans = samples,
    .sample_plans_cap = 4,
  };
  EXPECT(planner_plan(f.planner, &s, 1, 0, dst_strides, 3, &out) == DAMACY_OK);
  EXPECT(out.n_chunk_plans == 4);
  EXPECT(out.n_read_ops == 4);
  EXPECT(out.n_sample_plans == 1);

  // Chunks emitted row-major; their compressed_nbytes match per-file sizes.
  EXPECT(chunks[0].compressed_nbytes == 30);
  EXPECT(chunks[1].compressed_nbytes == 40);
  EXPECT(chunks[2].compressed_nbytes == 50);
  EXPECT(chunks[3].compressed_nbytes == 60);

  // Distinct chunk files: each read_op carries a distinct path.
  EXPECT(strcmp(reads[0].shard_path, "foo/c/0/0") == 0);
  EXPECT(strcmp(reads[1].shard_path, "foo/c/0/1") == 0);
  EXPECT(strcmp(reads[2].shard_path, "foo/c/1/0") == 0);
  EXPECT(strcmp(reads[3].shard_path, "foo/c/1/1") == 0);
  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT(reads[i].file_offset == 0);
    EXPECT(chunks[i].offset_in_read == 0);
  }

  fixture_destroy(&f);
  return 0;
}

// index_location: "start" — index sits at offset 0, payload follows.
// Verify the planner reads the right offsets (absolute file offsets in
// the index entries) and produces working chunk_plans.
static int
test_sharded_index_start(void)
{
  // 4 inner chunks; absolute offsets all clear of the index header
  // (index = 4*16+4 = 68 bytes). Use multiples of 256 for clarity.
  const uint64_t offsets[4] = { 256, 512, 768, 1024 };
  const uint64_t nbytes[4] = { 32, 32, 32, 32 };

  struct fixture f = { 0 };
  // Re-use fixture_init_with_json but write a start-mode shard.
  strcpy(f.root, "/tmp/damacy_planner_is_XXXXXX");
  EXPECT(mkdtemp(f.root));
  char p[256];
  snprintf(p, sizeof p, "%s/foo", f.root);
  EXPECT(mkdir(p, 0755) == 0);
  snprintf(p, sizeof p, "%s/foo/zarr.json", f.root);
  EXPECT(fixture_write_file(p, SHARDED_INDEX_START_ZARR_JSON) == 0);
  snprintf(p, sizeof p, "%s/foo/c", f.root);
  EXPECT(mkdir(p, 0755) == 0);
  snprintf(p, sizeof p, "%s/foo/c/0", f.root);
  EXPECT(mkdir(p, 0755) == 0);
  snprintf(p, sizeof p, "%s/foo/c/0/0", f.root);
  EXPECT(fixture_write_synthetic_shard_start(p, 4096, offsets, nbytes, 4) == 0);

  struct store_fs_config sc = { .root = f.root, .nthreads = 1 };
  f.store = store_fs_create(&sc);
  EXPECT(f.store);
  f.meta = zarr_meta_cache_create(f.store, 4);
  EXPECT(f.meta);
  f.shards = zarr_shard_cache_create(f.store, 4);
  EXPECT(f.shards);
  struct planner_config pcfg = {
    .meta_cache = f.meta,
    .shard_cache = f.shards,
    .page_alignment = PAGE,
  };
  EXPECT(planner_create(&pcfg, &f.planner) == DAMACY_OK);

  struct damacy_sample s = mk_sample("foo", 0, 4, 0, 8);
  int64_t dst_strides[3];
  mk_dst_strides_2d(1, 4, 8, dst_strides);
  struct read_op reads[8] = { 0 };
  struct chunk_plan chunks[8] = { 0 };
  struct sample_plan samples[4] = { 0 };
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 8,
    .chunk_plans = chunks,
    .chunk_plans_cap = 8,
    .sample_plans = samples,
    .sample_plans_cap = 4,
  };
  EXPECT(planner_plan(f.planner, &s, 1, 0, dst_strides, 3, &out) == DAMACY_OK);
  EXPECT(out.n_chunk_plans == 4);
  EXPECT(out.n_read_ops == 4);

  // All 4 chunks live inside the first PAGE, so each read_op page-aligns
  // to the same {file_offset=0, nbytes=PAGE}. The interesting per-chunk
  // values are offset_in_read (= absolute file offset, since page is 0)
  // and compressed_nbytes.
  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT(reads[i].file_offset == 0);
    EXPECT(reads[i].nbytes == PAGE);
    EXPECT(chunks[i].offset_in_read == offsets[i]);
    EXPECT(chunks[i].compressed_nbytes == nbytes[i]);
  }

  fixture_destroy(&f);
  return 0;
}

int
main(void)
{
  RUN(test_single_chunk_aligned);
  RUN(test_multi_chunk_partial);
  RUN(test_two_samples_indices);
  RUN(test_empty_chunk_becomes_fill);
  RUN(test_missing_shard_becomes_fill);
  RUN(test_fill_value_int16_neg1);
  RUN(test_fill_value_f32_nan);
  RUN(test_page_alignment);
  RUN(test_codec_id_blosc_lz4_rejected);
  RUN(test_codec_id_blosc_lz4hc_rejected);
  RUN(test_codec_id_blosc_zstd);
  RUN(test_codec_id_none);
  RUN(test_codec_id_blosc_unknown_cname);
  RUN(test_unsharded_single_chunk);
  RUN(test_unsharded_multi_chunk);
  RUN(test_sharded_index_start);
  log_info("all tests passed");
  return 0;
}
