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

// Same shape/layout as MINIMAL_ZARR_JSON but the inner codec is "blosc"
// with a configurable cname. %s is the cname ("lz4", "lz4hc", "zstd").
// Used by test_codec_id_blosc_* to verify parse_inner_codec resolves
// the inner cname into a CODEC_BLOSC_* tag and the planner propagates
// it to chunk_plan.codec_id.
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
  // Fixture zarr.json declares the inner codec as "zstd" → CODEC_ZSTD.
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

// Empty chunks inside a sample's AABB now fail (sparse zarrs unsupported).
static int
test_empty_chunks_fail(void)
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
  EXPECT(planner_plan(f.planner, &s, 1, 0, dst_strides, 3, &out) ==
         DAMACY_DECODE);

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

// Verify parse_inner_codec maps blosc cname → CODEC_BLOSC_* and the
// planner stamps the resolved tag onto every chunk_plan it emits.
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

static int
test_codec_id_blosc_lz4(void)
{
  return run_blosc_codec_id_case("lz4", CODEC_BLOSC_LZ4);
}

static int
test_codec_id_blosc_lz4hc(void)
{
  return run_blosc_codec_id_case("lz4hc", CODEC_BLOSC_LZ4);
}

static int
test_codec_id_blosc_zstd(void)
{
  return run_blosc_codec_id_case("zstd", CODEC_BLOSC_ZSTD);
}

int
main(void)
{
  RUN(test_single_chunk_aligned);
  RUN(test_multi_chunk_partial);
  RUN(test_two_samples_indices);
  RUN(test_empty_chunks_fail);
  RUN(test_page_alignment);
  RUN(test_codec_id_blosc_lz4);
  RUN(test_codec_id_blosc_lz4hc);
  RUN(test_codec_id_blosc_zstd);
  log_info("all tests passed");
  return 0;
}
