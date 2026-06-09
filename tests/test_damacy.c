// End-to-end smoke test for damacy: build a sharded zstd zarr v3 store
// via tests/write_zarr.py, push samples through the pipeline, copy the
// assembled batch back from device, verify byte-for-byte against the
// deterministic source content.
//
// Source content is data[i] = (linear_index + offset) masked to dtype
// (see tests/write_zarr.py). C tests reproduce expected values from
// the shape + offset without needing a shared RNG.
//
// Test cases:
//   test_full_array              — single zarr, samples_per_batch=1, full AABB
//   test_partial_crossing_chunks — single zarr, sub-window across 4 chunks
//   test_multi_batch             — single zarr, samples_per_batch=2, 3
//   pop-release
//                                  cycles with distinct AABBs per batch
//   test_multi_zarr              — two zarrs distinguished by fill offset;
//                                  batch with one sample from each
//   test_pipelined               — push 4 batches up front, pop 4 in a row
//                                  (drives W=2 + B=2 simultaneously)
//   test_lookahead_backpressure  — push past lookahead cap, expect AGAIN,
//                                  pop one, push remaining

#include "cuda_init.h"
#include "damacy.h"
#include "damacy_internal.h"
#include "fixture.h"
#include "platform/platform.h"
#include "spin_kernel.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

// Expected u16 value at a given (y, x) within a shape-(rows, cols)
// linear-fill zarr written with `--offset off`. Returned as float
// since the configured destination dtype is f32.
static float
expected_f32_from_u16_2d(int64_t y, int64_t x, int64_t cols, int64_t off)
{
  return (float)(uint16_t)((uint64_t)(y * cols + x + off) & 0xFFFFu);
}

static struct damacy_config
mk_cfg(const char* root, uint32_t samples_per_batch, int64_t sy, int64_t sx)
{
  (void)root;
  struct damacy_config c = {
    .samples_per_batch = samples_per_batch,
    .lookahead_samples = 2 * samples_per_batch,
    .dtype = DAMACY_F32,
    .sample_rank = 2,
    .device = -1,
  };
  c.tuning = damacy_tuning_defaults();
  c.tuning.n_io_threads = 1;
  c.tuning.metadata_io_concurrency = 1;
  // Cache floors (#134): n_*_cache >= lookahead_samples and
  // n_shard_index_cache >= lookahead_samples * max_shards_per_sample. These
  // fixtures use shard==array shape (1 shard/sample) so max_shards=1; 16
  // covers every lookahead used here (<= 8).
  c.tuning.n_array_meta_cache = 16;
  c.tuning.n_shard_index_cache = 16;
  c.tuning.n_chunk_layout_cache = 16;
  c.tuning.max_shards_per_sample = 1;
  c.tuning.max_gpu_memory_bytes = 1ull << 30;
  c.sample_shape[0] = sy;
  c.sample_shape[1] = sx;
  return c;
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
  if (cap < sizeof "/tmp/damacy_smoke_XXXXXX")
    return 1;
  strcpy(root, "/tmp/damacy_smoke_XXXXXX");
  return mkdtemp(root) ? 0 : 1;
}

// Push one sample, pop the resulting (size-1) batch, copy its device
// buffer to host. Caller-owned `out` must hold at least
// out_capacity_elements floats; the batch's element count is returned in
// *out_n_elements.
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
  EXPECT(info.ready_stream == (void*)d->wave_pool.stream_post);
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

// 4×8 zarr, full-AABB sample.
static int
test_full_array(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);
  int64_t shape[2] = { 4, 8 }, inner[2] = { 2, 4 }, shard[2] = { 4, 8 };
  EXPECT(fixture_write_zarr(p, shape, inner, shard, 2, "uint16", 0) == 0);

  struct damacy_config cfg = mk_cfg(root, 1, 4, 8);
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  float out[4 * 8] = { 0 };
  size_t got = 0;
  if (run_one(d, mk_sample(p, 0, 4, 0, 8), out, 4 * 8, &got))
    return 1;
  EXPECT(got == 4 * 8);
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 8; ++x)
      EXPECT(out[y * 8 + x] == expected_f32_from_u16_2d(y, x, 8, 0));

  damacy_destroy(d);
  fixture_rm_tree(root);
  return 0;
}

// Sub-window y[1,3) x[2,7) crosses the 2×2 inner-chunk grid; partial
// intersection per chunk. Output shape [1, 2, 5].
static int
test_partial_crossing_chunks(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);
  int64_t shape[2] = { 4, 8 }, inner[2] = { 2, 4 }, shard[2] = { 4, 8 };
  EXPECT(fixture_write_zarr(p, shape, inner, shard, 2, "uint16", 0) == 0);

  struct damacy_config cfg = mk_cfg(root, 1, 2, 5);
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  float out[2 * 5] = { 0 };
  size_t got = 0;
  if (run_one(d, mk_sample(p, 1, 3, 2, 7), out, 2 * 5, &got))
    return 1;
  EXPECT(got == 2 * 5);
  for (int y = 0; y < 2; ++y)
    for (int x = 0; x < 5; ++x)
      EXPECT(out[y * 5 + x] == expected_f32_from_u16_2d(1 + y, 2 + x, 8, 0));

  damacy_destroy(d);
  fixture_rm_tree(root);
  return 0;
}

// Three sequential batches off one zarr, each samples_per_batch=2 with
// distinct per-sample AABBs. Exercises pop-release-pop slot recycling
// (the failure mode step 5 needs to handle once batches overlap).
static int
test_multi_batch(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);
  int64_t shape[2] = { 8, 16 }, inner[2] = { 2, 4 }, shard[2] = { 8, 16 };
  EXPECT(fixture_write_zarr(p, shape, inner, shard, 2, "uint16", 0) == 0);

  struct damacy_config cfg = mk_cfg(root, 2, 4, 8);
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  // Each sample is 4×8; AABBs cover the four (4×8) quadrants of the
  // 8×16 array. Batch i takes quadrant pair {2i, 2i+1} (mod 4).
  const int64_t aabbs[6][4] = {
    { 0, 4, 0, 8 }, { 0, 4, 8, 16 }, // batch 0: top half
    { 4, 8, 0, 8 }, { 4, 8, 8, 16 }, // batch 1: bottom half
    { 0, 4, 0, 8 }, { 4, 8, 8, 16 }, // batch 2: cross-quadrant
  };

  for (int batch = 0; batch < 3; ++batch) {
    struct damacy_sample s[2] = {
      mk_sample(p,
                aabbs[batch * 2][0],
                aabbs[batch * 2][1],
                aabbs[batch * 2][2],
                aabbs[batch * 2][3]),
      mk_sample(p,
                aabbs[batch * 2 + 1][0],
                aabbs[batch * 2 + 1][1],
                aabbs[batch * 2 + 1][2],
                aabbs[batch * 2 + 1][3]),
    };
    struct damacy_sample_slice slice = { .beg = s, .end = s + 2 };
    struct damacy_push_result pr = damacy_push(d, slice);
    EXPECT(pr.status == DAMACY_OK);

    struct damacy_batch* b = NULL;
    EXPECT(damacy_pop(d, &b) == DAMACY_OK);
    struct damacy_batch_info info;
    damacy_batch_info(b, &info);
    EXPECT(info.rank == 3);
    EXPECT(info.shape[0] == 2);
    EXPECT(info.shape[1] == 4);
    EXPECT(info.shape[2] == 8);
    EXPECT(info.batch_id == (uint64_t)batch);

    float out[2 * 4 * 8] = { 0 };
    EXPECT(
      cudaMemcpy(out, info.device_ptr, sizeof out, cudaMemcpyDeviceToHost) ==
      cudaSuccess);

    for (int sample_idx = 0; sample_idx < 2; ++sample_idx) {
      int64_t y0 = aabbs[batch * 2 + sample_idx][0];
      int64_t x0 = aabbs[batch * 2 + sample_idx][2];
      for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 8; ++x) {
          float got = out[sample_idx * 32 + y * 8 + x];
          float want = expected_f32_from_u16_2d(y0 + y, x0 + x, 16, 0);
          EXPECT(got == want);
        }
      }
    }

    damacy_release(d, b);
  }

  damacy_destroy(d);
  fixture_rm_tree(root);
  return 0;
}

// One batch with samples drawn from two different zarrs. The two zarrs
// share dtype + sample shape but differ in fill offset, so the popped
// batch should contain one slot of "a" content and one of "b" content
// — verifying meta+shard cache routing across uris.
static int
test_multi_zarr(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char pa[256], pb[256];
  snprintf(pa, sizeof pa, "%s/a", root);
  snprintf(pb, sizeof pb, "%s/b", root);
  int64_t shape[2] = { 4, 8 }, inner[2] = { 2, 4 }, shard[2] = { 4, 8 };
  EXPECT(fixture_write_zarr(pa, shape, inner, shard, 2, "uint16", 0) == 0);
  EXPECT(fixture_write_zarr(pb, shape, inner, shard, 2, "uint16", 1000) == 0);

  struct damacy_config cfg = mk_cfg(root, 2, 4, 8);
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  struct damacy_sample s[2] = {
    mk_sample(pa, 0, 4, 0, 8),
    mk_sample(pb, 0, 4, 0, 8),
  };
  struct damacy_sample_slice slice = { .beg = s, .end = s + 2 };
  struct damacy_push_result pr = damacy_push(d, slice);
  EXPECT(pr.status == DAMACY_OK);

  struct damacy_batch* b = NULL;
  EXPECT(damacy_pop(d, &b) == DAMACY_OK);
  struct damacy_batch_info info;
  damacy_batch_info(b, &info);
  EXPECT(info.shape[0] == 2);
  EXPECT(info.shape[1] == 4);
  EXPECT(info.shape[2] == 8);

  float out[2 * 4 * 8] = { 0 };
  EXPECT(cudaMemcpy(out, info.device_ptr, sizeof out, cudaMemcpyDeviceToHost) ==
         cudaSuccess);
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 8; ++x) {
      EXPECT(out[0 * 32 + y * 8 + x] == expected_f32_from_u16_2d(y, x, 8, 0));
      EXPECT(out[1 * 32 + y * 8 + x] ==
             expected_f32_from_u16_2d(y, x, 8, 1000));
    }
  }
  damacy_release(d, b);

  damacy_destroy(d);
  fixture_rm_tree(root);
  return 0;
}

static int
test_incremental_batch_fill(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);
  int64_t shape[2] = { 8, 16 }, inner[2] = { 2, 4 }, shard[2] = { 8, 16 };
  EXPECT(fixture_write_zarr(p, shape, inner, shard, 2, "uint16", 0) == 0);

  struct damacy_config cfg = mk_cfg(root, 2, 4, 8);
  cfg.lookahead_samples = 2;
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  struct damacy_sample s0 = mk_sample(p, 0, 4, 0, 8);
  struct damacy_sample s1 = mk_sample(p, 4, 8, 8, 16);
  EXPECT(damacy_push(d, (struct damacy_sample_slice){ &s0, &s0 + 1 }).status ==
         DAMACY_OK);
  platform_sleep_ns(20000000);
  EXPECT(damacy_push(d, (struct damacy_sample_slice){ &s1, &s1 + 1 }).status ==
         DAMACY_OK);

  struct damacy_batch* b = NULL;
  EXPECT(damacy_pop(d, &b) == DAMACY_OK);
  struct damacy_batch_info info;
  damacy_batch_info(b, &info);
  EXPECT(info.batch_id == 0);
  EXPECT(info.shape[0] == 2);
  EXPECT(info.shape[1] == 4);
  EXPECT(info.shape[2] == 8);

  float out[2 * 4 * 8] = { 0 };
  EXPECT(cudaMemcpy(out, info.device_ptr, sizeof out, cudaMemcpyDeviceToHost) ==
         cudaSuccess);
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 8; ++x) {
      EXPECT(out[y * 8 + x] == expected_f32_from_u16_2d(y, x, 16, 0));
      EXPECT(out[32 + y * 8 + x] ==
             expected_f32_from_u16_2d(4 + y, 8 + x, 16, 0));
    }
  }
  damacy_release(d, b);

  damacy_destroy(d);
  fixture_rm_tree(root);
  return 0;
}

// Mixed source dtypes within one batch: one u8 zarr and one u16 zarr,
// both cast to the configured f32 destination. Verifies the per-block
// src_dtype dispatch in the assemble kernel.
static int
test_heterogeneous_dtype(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char pa[256], pb[256];
  snprintf(pa, sizeof pa, "%s/a", root);
  snprintf(pb, sizeof pb, "%s/b", root);
  int64_t shape[2] = { 4, 8 }, inner[2] = { 2, 4 }, shard[2] = { 4, 8 };
  EXPECT(fixture_write_zarr(pa, shape, inner, shard, 2, "uint8", 0) == 0);
  EXPECT(fixture_write_zarr(pb, shape, inner, shard, 2, "uint16", 1000) == 0);

  struct damacy_config cfg = mk_cfg(root, 2, 4, 8);
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  struct damacy_sample s[2] = {
    mk_sample(pa, 0, 4, 0, 8),
    mk_sample(pb, 0, 4, 0, 8),
  };
  struct damacy_sample_slice slice = { .beg = s, .end = s + 2 };
  struct damacy_push_result pr = damacy_push(d, slice);
  EXPECT(pr.status == DAMACY_OK);

  struct damacy_batch* b = NULL;
  EXPECT(damacy_pop(d, &b) == DAMACY_OK);
  struct damacy_batch_info info;
  damacy_batch_info(b, &info);
  EXPECT(info.shape[0] == 2);
  EXPECT(info.shape[1] == 4);
  EXPECT(info.shape[2] == 8);
  EXPECT(info.dtype == DAMACY_F32);

  float out[2 * 4 * 8] = { 0 };
  EXPECT(cudaMemcpy(out, info.device_ptr, sizeof out, cudaMemcpyDeviceToHost) ==
         cudaSuccess);
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 8; ++x) {
      // sample 0: u8 source, fill offset 0 → value masked to u8 range.
      uint8_t want_u8 = (uint8_t)((uint64_t)(y * 8 + x + 0) & 0xFFu);
      EXPECT(out[0 * 32 + y * 8 + x] == (float)want_u8);
      // sample 1: u16 source, fill offset 1000.
      EXPECT(out[1 * 32 + y * 8 + x] ==
             expected_f32_from_u16_2d(y, x, 8, 1000));
    }
  }
  damacy_release(d, b);

  damacy_destroy(d);
  fixture_rm_tree(root);
  return 0;
}

// Push 4 batches' worth of samples up front, then pop 4 in a row. The
// scheduler must overlap W=2 waves and use both B=2 batch slots
// simultaneously. Verifies pipelined throughput + correctness across
// multiple in-flight batches.
static int
test_pipelined(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);
  int64_t shape[2] = { 8, 16 }, inner[2] = { 2, 4 }, shard[2] = { 8, 16 };
  EXPECT(fixture_write_zarr(p, shape, inner, shard, 2, "uint16", 0) == 0);

  struct damacy_config cfg = mk_cfg(root, 2, 4, 8);
  // Bump lookahead so we can hold 4 batches' worth of samples up front.
  cfg.lookahead_samples = 8;
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  // Each sample is 4×8; 4 batches × 2 samples = 8 samples covering 8
  // non-overlapping (4×8) regions of the 8×16 array (only 4 unique
  // positions, so we cycle through them twice).
  const int64_t aabbs[8][4] = {
    { 0, 4, 0, 8 },  { 0, 4, 8, 16 }, // batch 0
    { 4, 8, 0, 8 },  { 4, 8, 8, 16 }, // batch 1
    { 0, 4, 8, 16 }, { 4, 8, 0, 8 },  // batch 2
    { 0, 4, 0, 8 },  { 4, 8, 8, 16 }, // batch 3
  };
  struct damacy_sample samples[8];
  for (int i = 0; i < 8; ++i)
    samples[i] =
      mk_sample(p, aabbs[i][0], aabbs[i][1], aabbs[i][2], aabbs[i][3]);

  // Push all 8 in one shot.
  struct damacy_sample_slice slice = { .beg = samples, .end = samples + 8 };
  struct damacy_push_result pr = damacy_push(d, slice);
  EXPECT(pr.status == DAMACY_OK);
  EXPECT(pr.unconsumed.beg == pr.unconsumed.end);

  // Pop 4 batches in order, verify each one's contents.
  for (int batch = 0; batch < 4; ++batch) {
    struct damacy_batch* b = NULL;
    EXPECT(damacy_pop(d, &b) == DAMACY_OK);
    struct damacy_batch_info info;
    damacy_batch_info(b, &info);
    EXPECT(info.batch_id == (uint64_t)batch);
    EXPECT(info.shape[0] == 2);
    EXPECT(info.shape[1] == 4);
    EXPECT(info.shape[2] == 8);

    float out[2 * 4 * 8] = { 0 };
    EXPECT(
      cudaMemcpy(out, info.device_ptr, sizeof out, cudaMemcpyDeviceToHost) ==
      cudaSuccess);

    for (int sample_idx = 0; sample_idx < 2; ++sample_idx) {
      int64_t y0 = aabbs[batch * 2 + sample_idx][0];
      int64_t x0 = aabbs[batch * 2 + sample_idx][2];
      for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 8; ++x) {
          float got = out[sample_idx * 32 + y * 8 + x];
          float want = expected_f32_from_u16_2d(y0 + y, x0 + x, 16, 0);
          EXPECT(got == want);
        }
      }
    }
    damacy_release(d, b);
  }

  damacy_destroy(d);
  fixture_rm_tree(root);
  return 0;
}

// Push past the lookahead cap; expect DAMACY_AGAIN with a non-empty
// unconsumed suffix; pop one batch; push the suffix; verify we get the
// rest. Verifies push-side backpressure surfacing through to the user.
static int
test_lookahead_backpressure(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);
  int64_t shape[2] = { 4, 8 }, inner[2] = { 2, 4 }, shard[2] = { 4, 8 };
  EXPECT(fixture_write_zarr(p, shape, inner, shard, 2, "uint16", 0) == 0);

  // samples_per_batch=1, lookahead_samples=2 → lookahead cap of 2 samples.
  struct damacy_config cfg = mk_cfg(root, 1, 4, 8);
  cfg.lookahead_samples = 2;
  cfg.debug.metadata_latency.baseline_ns = 50ull * 1000ull * 1000ull;
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  // Try to push 3 samples; only 2 should land before AGAIN.
  struct damacy_sample samples[3] = {
    mk_sample(p, 0, 4, 0, 8),
    mk_sample(p, 0, 4, 0, 8),
    mk_sample(p, 0, 4, 0, 8),
  };
  struct damacy_sample_slice slice = { .beg = samples, .end = samples + 3 };
  struct damacy_push_result pr = damacy_push(d, slice);
  EXPECT(pr.status == DAMACY_AGAIN);
  EXPECT(pr.unconsumed.beg == samples + 2);
  EXPECT(pr.unconsumed.end == samples + 3);

  // Pop one batch to free up a slot (and a lookahead spot).
  struct damacy_batch* b = NULL;
  EXPECT(damacy_pop(d, &b) == DAMACY_OK);
  damacy_release(d, b);

  // Now push the rest — should succeed.
  pr = damacy_push(d, pr.unconsumed);
  EXPECT(pr.status == DAMACY_OK);
  EXPECT(pr.unconsumed.beg == pr.unconsumed.end);

  // Drain the remaining batches.
  EXPECT(damacy_pop(d, &b) == DAMACY_OK);
  damacy_release(d, b);
  EXPECT(damacy_pop(d, &b) == DAMACY_OK);
  damacy_release(d, b);

  damacy_destroy(d);
  fixture_rm_tree(root);
  return 0;
}

// Missing-shard end-to-end: write a normal zarr, delete its shard
// file, and verify the popped batch is full of fill_value (zero in this
// case because write_zarr.py writes fill_value=0 by default).
static int
test_missing_shard_fills(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);
  int64_t shape[2] = { 4, 8 }, inner[2] = { 2, 4 }, shard[2] = { 4, 8 };
  EXPECT(fixture_write_zarr(p, shape, inner, shard, 2, "uint16", 0) == 0);

  // Delete the shard file so reads return zero bytes; planner emits fill.
  char shard_path[512];
  snprintf(shard_path, sizeof shard_path, "%s/foo/c/0/0", root);
  EXPECT(unlink(shard_path) == 0);

  struct damacy_config cfg = mk_cfg(root, 1, 4, 8);
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  float out[4 * 8] = { 0 };
  size_t got = 0;
  if (run_one(d, mk_sample(p, 0, 4, 0, 8), out, 4 * 8, &got))
    return 1;
  EXPECT(got == 4 * 8);
  // fill_value=0 → every voxel is 0.0f.
  for (int i = 0; i < 4 * 8; ++i)
    EXPECT(out[i] == 0.0f);

  damacy_destroy(d);
  fixture_rm_tree(root);
  return 0;
}

// Same as test_missing_shard_fills but with a non-zero fill_value. The
// dst buffer is zero-initialised so this would silently pass if the
// fill path were broken; the explicit non-zero value catches regressions
// where the cast/load path produces 0.0f. fixture_write_zarr doesn't
// expose fill_value, so the generated zarr.json is rewritten in-place.
static int
test_missing_shard_fills_nonzero(void)
{
  static const char* ZARR_JSON_FILL_42 =
    "{"
    "\"zarr_format\":3,"
    "\"node_type\":\"array\","
    "\"shape\":[4,8],"
    "\"data_type\":\"uint16\","
    "\"chunk_grid\":{\"name\":\"regular\",\"configuration\":{"
    "\"chunk_shape\":[4,8]}},"
    "\"chunk_key_encoding\":{\"name\":\"default\",\"configuration\":{"
    "\"separator\":\"/\"}},"
    "\"fill_value\":42,"
    "\"codecs\":[{\"name\":\"sharding_indexed\",\"configuration\":{"
    "\"chunk_shape\":[2,4],"
    "\"codecs\":[{\"name\":\"bytes\",\"configuration\":{\"endian\":\"little\"}}"
    ","
    "{\"name\":\"zstd\",\"configuration\":{\"level\":3,\"checksum\":false}}],"
    "\"index_codecs\":[{\"name\":\"bytes\",\"configuration\":{"
    "\"endian\":\"little\"}},{\"name\":\"crc32c\"}],"
    "\"index_location\":\"end\"}}]"
    "}";

  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);
  int64_t shape[2] = { 4, 8 }, inner[2] = { 2, 4 }, shard[2] = { 4, 8 };
  EXPECT(fixture_write_zarr(p, shape, inner, shard, 2, "uint16", 0) == 0);

  // Overwrite zarr.json with a non-zero fill_value.
  char meta_path[512];
  snprintf(meta_path, sizeof meta_path, "%s/foo/zarr.json", root);
  EXPECT(fixture_write_file(meta_path, ZARR_JSON_FILL_42) == 0);

  // Delete the shard file: every chunk is now absent → fill_value.
  char shard_path[512];
  snprintf(shard_path, sizeof shard_path, "%s/foo/c/0/0", root);
  EXPECT(unlink(shard_path) == 0);

  struct damacy_config cfg = mk_cfg(root, 1, 4, 8);
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  float out[4 * 8] = { 0 };
  size_t got = 0;
  if (run_one(d, mk_sample(p, 0, 4, 0, 8), out, 4 * 8, &got))
    return 1;
  EXPECT(got == 4 * 8);
  for (int i = 0; i < 4 * 8; ++i)
    EXPECT(out[i] == 42.0f);

  damacy_destroy(d);
  fixture_rm_tree(root);
  return 0;
}

// Engineered race for damacy_release_event.
//
// The side stream runs a GPU-side spin (clock64 loop) and records an
// event after it. release(event) installs cuStreamWaitEvent on damacy's
// stream_post; the third pop's assemble can't run until that event fires.
//
// To keep the spin from blocking stream_post via hardware-queue
// contention on consumer GPUs (Blackwell schedules a long-running kernel
// ahead of newcomers regardless of stream), we put the side stream at
// the LOW priority slot and damacy's streams stay at default — the GPU
// then time-slices in favor of stream_post and the spin can't starve it.
static int
test_release_event_blocks_assemble(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/z", root);
  int64_t shape[2] = { 4, 8 }, inner[2] = { 4, 8 }, shard[2] = { 4, 8 };
  EXPECT(fixture_write_zarr(p, shape, inner, shard, 2, "uint16", 0) == 0);

  struct damacy_config cfg = mk_cfg(root, 1, 4, 8);
  cfg.lookahead_samples = 3; // pool has 2 slots; 3 pops forces reuse.
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  struct damacy_sample s = mk_sample(p, 0, 4, 0, 8);
  struct damacy_sample s_arr[3] = { s, s, s };
  struct damacy_sample_slice slice = { .beg = s_arr, .end = s_arr + 3 };
  EXPECT(damacy_push(d, slice).status == DAMACY_OK);

  int prio_lo = 0, prio_hi = 0;
  EXPECT(cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi) == cudaSuccess);
  if (prio_lo == prio_hi) {
    log_warn("stream priorities are degenerate (lo==hi==%d); skipping "
             "test_release_event_blocks_assemble — the spin kernel would "
             "share priority with stream_post and could pass for the wrong "
             "reason",
             prio_lo);
    damacy_destroy(d);
    fixture_rm_tree(root);
    return 0;
  }
  cudaStream_t side = NULL;
  EXPECT(cudaStreamCreateWithPriority(&side, cudaStreamNonBlocking, prio_lo) ==
         cudaSuccess);

  struct damacy_batch* b = NULL;
  EXPECT(damacy_pop(d, &b) == DAMACY_OK);

  // ~100 ms of spin at SM-clock rates (1–2 GHz on modern parts).
  const int64_t cycles = 200LL * 1000 * 1000;
  EXPECT(test_spin_launch(side, cycles) == cudaSuccess);
  cudaEvent_t ev = NULL;
  EXPECT(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) == cudaSuccess);
  EXPECT(cudaEventRecord(ev, side) == cudaSuccess);

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  EXPECT(damacy_release_event(d, b, ev) == DAMACY_OK);
  EXPECT(cudaEventDestroy(ev) == cudaSuccess);

  EXPECT(damacy_pop(d, &b) == DAMACY_OK);
  damacy_release(d, b);
  EXPECT(damacy_pop(d, &b) == DAMACY_OK); // blocks until ev fires
  damacy_release(d, b);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double elapsed_ms = (double)(t1.tv_sec - t0.tv_sec) * 1000.0 +
                      (double)(t1.tv_nsec - t0.tv_nsec) / 1e6;
  log_info("release+3-pop window: %.1f ms (expect >= 50)", elapsed_ms);
  EXPECT(elapsed_ms >= 50.0);

  EXPECT(cudaStreamSynchronize(side) == cudaSuccess);
  cudaStreamDestroy(side);
  damacy_destroy(d);
  fixture_rm_tree(root);
  return 0;
}

// damacy_release_event(d, b, NULL) should behave exactly like
// damacy_release: slot returns to the pool, next pop works.
static int
test_release_event_null_falls_back(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);
  int64_t shape[2] = { 4, 8 }, inner[2] = { 4, 8 }, shard[2] = { 4, 8 };
  EXPECT(fixture_write_zarr(p, shape, inner, shard, 2, "uint16", 0) == 0);

  struct damacy_config cfg = mk_cfg(root, 1, 4, 8);
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  struct damacy_sample s = mk_sample(p, 0, 4, 0, 8);
  struct damacy_sample_slice slice = { .beg = &s, .end = &s + 1 };
  EXPECT(damacy_push(d, slice).status == DAMACY_OK);

  struct damacy_batch* b = NULL;
  EXPECT(damacy_pop(d, &b) == DAMACY_OK);
  EXPECT(damacy_release_event(d, b, NULL) == DAMACY_OK);

  // Push + pop again to confirm the slot recycled cleanly.
  EXPECT(damacy_push(d, slice).status == DAMACY_OK);
  EXPECT(damacy_pop(d, &b) == DAMACY_OK);
  damacy_release(d, b);

  damacy_destroy(d);
  fixture_rm_tree(root);
  return 0;
}

int
main(void)
{
  EXPECT(cuda_init_primary() == 0);
  RUN(test_full_array);
  RUN(test_partial_crossing_chunks);
  RUN(test_multi_batch);
  RUN(test_multi_zarr);
  RUN(test_incremental_batch_fill);
  RUN(test_heterogeneous_dtype);
  RUN(test_pipelined);
  RUN(test_lookahead_backpressure);
  RUN(test_missing_shard_fills);
  RUN(test_missing_shard_fills_nonzero);
  RUN(test_release_event_blocks_assemble);
  RUN(test_release_event_null_falls_back);
  log_info("all tests passed");
  return 0;
}
