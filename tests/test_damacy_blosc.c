// End-to-end smoke test for damacy with blosc-encoded zarr inputs.
// Mirrors test_damacy.c but drives the blosc1 GPU pipeline path
// (parse + scan + emit + nvcomp + (un)shuffle + assemble).
//
// Test cases:
//   test_full_array_blosc_zstd          — single zarr, blosc(zstd) codec
//   test_full_array_blosc_lz4           — single zarr, blosc(lz4) codec
//   test_partial_crossing_chunks_blosc  — sub-window across 4 blosc-lz4 chunks
//   test_four_codecs_mixed_batch        — none + zstd + blosc-zstd + blosc-lz4
//   test_multi_wave_per_batch           — one batch that exceeds per-wave
//                                         caps; pipeline must split it
//                                         into ≥2 waves of the same batch

#include "cuda_init.h"
#include "damacy.h"
#include "fixture.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

static float
expected_f32_from_u16_2d(int64_t y, int64_t x, int64_t cols, int64_t off)
{
  return (float)(uint16_t)((uint64_t)(y * cols + x + off) & 0xFFFFu);
}

static struct damacy_config
mk_cfg(const char* root, uint32_t batch_size)
{
  return (struct damacy_config){
    .store_root = root,
    .batch_size = batch_size,
    .lookahead_batches = 2,
    .n_io_threads = 1,
    .host_buffer_bytes = 1ull << 20,
    .device_buffer_bytes = 1ull << 20,
    .n_zarrs_meta_cache = 4,
    .n_shards_meta_cache = 4,
    .dtype = DAMACY_F32,
  };
}

static struct damacy_sample
mk_sample(const char* uri, int64_t y0, int64_t y1, int64_t x0, int64_t x1)
{
  struct damacy_sample s = { .uri = uri, .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = y0, .end = y1 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = x0, .end = x1 };
  return s;
}

static int
mkdtemp_root(char* root, size_t cap)
{
  if (cap < sizeof "/tmp/damacy_blosc_XXXXXX")
    return 1;
  strcpy(root, "/tmp/damacy_blosc_XXXXXX");
  return mkdtemp(root) ? 0 : 1;
}

// Push one sample, pop the resulting size-1 batch, copy to host.
static int
run_one(struct damacy* d,
        struct damacy_sample s,
        float* out,
        size_t out_capacity_elements,
        size_t* out_n_elements)
{
  struct damacy_sample_slice slice = { .beg = &s, .end = &s + 1 };
  struct damacy_push_result pr = damacy_push(d, slice);
  EXPECT(pr.status == DAMACY_OK);

  struct damacy_batch* b = NULL;
  EXPECT(damacy_pop(d, &b) == DAMACY_OK);

  struct damacy_batch_info info;
  damacy_batch_info(b, &info);
  EXPECT(info.rank == 3);
  EXPECT(info.dtype == DAMACY_F32);
  EXPECT(info.shape[0] == 1);
  size_t n_elements = (size_t)info.shape[1] * (size_t)info.shape[2];
  EXPECT(n_elements <= out_capacity_elements);
  EXPECT(cudaMemcpy(out,
                    info.device_ptr,
                    n_elements * sizeof(float),
                    cudaMemcpyDeviceToHost) == cudaSuccess);
  *out_n_elements = n_elements;
  damacy_release(d, b);
  return 0;
}

// Drive the full damacy pipeline against a zarr written with `codec`,
// reading the entire array as a single sample and verifying byte content.
static int
run_full_array(const char* codec)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);
  // 16×32 uint16 = 1 KB; inner 8×16 = 256 B per chunk; 4 chunks per
  // shard. Big enough that blosc compresses with shuffle+lz4/zstd.
  int64_t shape[2] = { 16, 32 }, inner[2] = { 8, 16 }, shard[2] = { 16, 32 };
  EXPECT(fixture_write_zarr_codec(
           p, shape, inner, shard, 2, "uint16", 0, codec) == 0);

  struct damacy_config cfg = mk_cfg(root, 1);
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  float out[16 * 32] = { 0 };
  size_t got = 0;
  if (run_one(d, mk_sample("foo", 0, 16, 0, 32), out, 16 * 32, &got))
    return 1;
  EXPECT(got == 16 * 32);
  for (int y = 0; y < 16; ++y)
    for (int x = 0; x < 32; ++x)
      EXPECT(out[y * 32 + x] == expected_f32_from_u16_2d(y, x, 32, 0));

  damacy_destroy(d);
  fixture_rm_tree(root);
  return 0;
}

static int
test_full_array_blosc_zstd(void)
{
  return run_full_array("blosc-zstd");
}

static int
test_full_array_blosc_lz4(void)
{
  return run_full_array("blosc-lz4");
}

// Sub-window y[3,11) x[5,27) crosses the 2x2 inner-chunk grid;
// exercises the blosc-lz4 + shuffle path with partial intersections.
static int
test_partial_crossing_chunks_blosc(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);
  int64_t shape[2] = { 16, 32 }, inner[2] = { 8, 16 }, shard[2] = { 16, 32 };
  EXPECT(fixture_write_zarr_codec(
           p, shape, inner, shard, 2, "uint16", 0, "blosc-lz4") == 0);

  struct damacy_config cfg = mk_cfg(root, 1);
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  const int H = 8, W = 22;
  float out[8 * 22] = { 0 };
  size_t got = 0;
  if (run_one(d, mk_sample("foo", 3, 11, 5, 27), out, H * W, &got))
    return 1;
  EXPECT(got == (size_t)(H * W));
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      EXPECT(out[y * W + x] == expected_f32_from_u16_2d(3 + y, 5 + x, 32, 0));

  damacy_destroy(d);
  fixture_rm_tree(root);
  return 0;
}

// Four zarrs (none, zstd, blosc-zstd, blosc-lz4), one sample each in a
// batch_size=4 batch. All four codecs route through the unified blosc1
// GPU pipeline; the wave dispatches to memcpy + nvcomp_zstd + nvcomp_lz4
// in parallel.
static int
test_four_codecs_mixed_batch(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  const char* names[4] = { "n", "z", "bz", "bl" };
  const char* codecs[4] = { "none", "zstd", "blosc-zstd", "blosc-lz4" };
  const int64_t offsets[4] = { 0, 1000, 2000, 3000 };
  int64_t shape[2] = { 16, 32 }, inner[2] = { 8, 16 }, shard[2] = { 16, 32 };
  for (int i = 0; i < 4; ++i) {
    char p[256];
    snprintf(p, sizeof p, "%s/%s", root, names[i]);
    EXPECT(fixture_write_zarr_codec(
             p, shape, inner, shard, 2, "uint16", offsets[i], codecs[i]) == 0);
  }

  struct damacy_config cfg = mk_cfg(root, 4);
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  struct damacy_sample s[4];
  for (int i = 0; i < 4; ++i)
    s[i] = mk_sample(names[i], 0, 16, 0, 32);
  struct damacy_sample_slice slice = { .beg = s, .end = s + 4 };
  struct damacy_push_result pr = damacy_push(d, slice);
  EXPECT(pr.status == DAMACY_OK);

  struct damacy_batch* b = NULL;
  EXPECT(damacy_pop(d, &b) == DAMACY_OK);
  struct damacy_batch_info info;
  damacy_batch_info(b, &info);
  EXPECT(info.shape[0] == 4);
  EXPECT(info.shape[1] == 16);
  EXPECT(info.shape[2] == 32);

  float out[4 * 16 * 32] = { 0 };
  EXPECT(cudaMemcpy(out, info.device_ptr, sizeof out, cudaMemcpyDeviceToHost) ==
         cudaSuccess);
  for (int i = 0; i < 4; ++i)
    for (int y = 0; y < 16; ++y)
      for (int x = 0; x < 32; ++x)
        EXPECT(out[i * 16 * 32 + y * 32 + x] ==
               expected_f32_from_u16_2d(y, x, 32, offsets[i]));
  damacy_release(d, b);

  // Sub-stage metrics: blosc1 GPU pipeline must have stamped each
  // wave's parse and post events. zstd and lz4 are codec-conditional
  // but this batch contains both.
  struct damacy_stats st;
  damacy_stats_get(d, &st);
  EXPECT(st.decompress_parse.count > 0);
  EXPECT(st.decompress_zstd.count > 0);
  EXPECT(st.decompress_lz4.count > 0);
  EXPECT(st.decompress_post.count > 0);

  damacy_destroy(d);
  fixture_rm_tree(root);
  return 0;
}

// One batch whose total bytes exceed per-wave caps, forcing the
// scheduler to split it across multiple waves of the same batch slot.
// Verifies (a) byte content end-to-end across the split, and (b) that
// at least two waves fired for the single batch (waves_emitted >= 2).
//
// host/dev buffers are intentionally small so each wave only fits a
// fraction of the batch's chunks. With four 16×32 u16 zarrs each split
// into 4 inner chunks (16 chunks total at ~256 B raw + ~80–160 B
// compressed each), 2 KiB per wave forces at least two waves.
static int
test_multi_wave_per_batch(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  // Mix of codecs so the multi-wave path also exercises mixed-codec
  // routing within and across waves.
  const char* names[4] = { "a", "b", "c", "d" };
  const char* codecs[4] = { "blosc-lz4", "blosc-zstd", "zstd", "blosc-lz4" };
  const int64_t offsets[4] = { 0, 1000, 2000, 3000 };
  int64_t shape[2] = { 16, 32 }, inner[2] = { 8, 16 }, shard[2] = { 16, 32 };
  for (int i = 0; i < 4; ++i) {
    char p[256];
    snprintf(p, sizeof p, "%s/%s", root, names[i]);
    EXPECT(fixture_write_zarr_codec(
             p, shape, inner, shard, 2, "uint16", offsets[i], codecs[i]) == 0);
  }

  // peel_wave's per-chunk read_op is page-aligned (typically 4 KiB), so
  // host_slab_cap = host_buffer_bytes/2 must fit at least one full
  // page. We pick host = 64 KiB → 32 KiB/wave → ~8 chunks/wave; the
  // 16-chunk batch spills into 2 waves of the same batch slot.
  struct damacy_config cfg = {
    .store_root = root,
    .batch_size = 4,
    .lookahead_batches = 2,
    .n_io_threads = 1,
    .host_buffer_bytes = 64ull << 10,
    .device_buffer_bytes = 64ull << 10,
    .n_zarrs_meta_cache = 4,
    .n_shards_meta_cache = 4,
    .dtype = DAMACY_F32,
  };
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  struct damacy_sample s[4];
  for (int i = 0; i < 4; ++i)
    s[i] = mk_sample(names[i], 0, 16, 0, 32);
  struct damacy_sample_slice slice = { .beg = s, .end = s + 4 };
  struct damacy_push_result pr = damacy_push(d, slice);
  EXPECT(pr.status == DAMACY_OK);

  struct damacy_batch* b = NULL;
  EXPECT(damacy_pop(d, &b) == DAMACY_OK);
  struct damacy_batch_info info;
  damacy_batch_info(b, &info);
  EXPECT(info.shape[0] == 4);
  EXPECT(info.shape[1] == 16);
  EXPECT(info.shape[2] == 32);

  float out[4 * 16 * 32] = { 0 };
  EXPECT(cudaMemcpy(out, info.device_ptr, sizeof out, cudaMemcpyDeviceToHost) ==
         cudaSuccess);
  for (int i = 0; i < 4; ++i)
    for (int y = 0; y < 16; ++y)
      for (int x = 0; x < 32; ++x)
        EXPECT(out[i * 16 * 32 + y * 32 + x] ==
               expected_f32_from_u16_2d(y, x, 32, offsets[i]));
  damacy_release(d, b);

  // Splitting evidence: the four-sample batch produced at least two
  // waves (a single-wave batch would set waves_emitted == 1).
  struct damacy_stats st;
  damacy_stats_get(d, &st);
  EXPECT(st.batches_emitted == 1);
  EXPECT(st.waves_emitted >= 2);
  EXPECT(st.chunks_dispatched == 16); // 4 inner chunks × 4 samples

  damacy_destroy(d);
  fixture_rm_tree(root);
  return 0;
}

int
main(void)
{
  EXPECT(cuda_init_primary() == 0);
  RUN(test_full_array_blosc_zstd);
  RUN(test_full_array_blosc_lz4);
  RUN(test_partial_crossing_chunks_blosc);
  RUN(test_four_codecs_mixed_batch);
  RUN(test_multi_wave_per_batch);
  log_info("all tests passed");
  return 0;
}
