// End-to-end smoke test for damacy with blosc-encoded zarr inputs.
// Mirrors test_damacy.c but drives the blosc1 GPU pipeline path
// (parse + scan + emit + nvcomp + (un)shuffle + assemble).
//
// Test cases:
//   test_full_array_blosc_zstd          — single zarr, blosc(zstd) codec
//   test_partial_crossing_chunks_blosc  — sub-window across 4 blosc-zstd chunks
//   test_three_codecs_mixed_batch        — none + zstd + blosc-zstd × 2
//   test_multi_wave_per_batch           — one batch that exceeds per-wave
//                                         caps; pipeline must split it
//                                         into ≥2 waves of the same batch
//   test_wave_grows_substream_cap       — single wave with >32 chunks forces
//                                         per-wave fanout + decoder-scratch
//                                         grow past the 1024 initial floor
//   test_grow_inside_tight_budget       — tight budget + grow load;
//                                         resolver's worst-case reservation
//                                         keeps the grow inside the cap
//   test_layout_probe_avoids_decoder_grow
//                                       — with the planner's chunk_layout
//                                         probe, need_zsubs is computed
//                                         from probed nblocks; the
//                                         single-block fixture stays
//                                         under the initial cap and the
//                                         decoder grow never fires
//   test_batch_pool_rejected_at_inflated_committed
//                                       — inflate fully to the cap;
//                                         batch-output pool allocation fails
//                                         before the grow paths run

#include "cuda_init.h"
#include "damacy.h"
#include "fixture.h"
#include "store/store.h"
#include "store/store_fs_gds.h"

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
  (void)root;
  return (struct damacy_config){
    .batch_size = batch_size,
    .lookahead_batches = 2,
    .dtype = DAMACY_F32,
    .device = -1,
    .tuning = {
      .n_io_threads = 1,
      .n_zarrs_meta_cache = 4,
      .n_shards_meta_cache = 4,
    },
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
  // shard. Big enough that blosc compresses with shuffle+zstd.
  int64_t shape[2] = { 16, 32 }, inner[2] = { 8, 16 }, shard[2] = { 16, 32 };
  EXPECT(fixture_write_zarr_codec(
           p, shape, inner, shard, 2, "uint16", 0, codec) == 0);

  struct damacy_config cfg = mk_cfg(root, 1);
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  float out[16 * 32] = { 0 };
  size_t got = 0;
  if (run_one(d, mk_sample(p, 0, 16, 0, 32), out, 16 * 32, &got))
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

// Sub-window y[3,11) x[5,27) crosses the 2x2 inner-chunk grid;
// exercises the blosc-zstd + shuffle path with partial intersections.
static int
test_partial_crossing_chunks_blosc(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);
  int64_t shape[2] = { 16, 32 }, inner[2] = { 8, 16 }, shard[2] = { 16, 32 };
  EXPECT(fixture_write_zarr_codec(
           p, shape, inner, shard, 2, "uint16", 0, "blosc-zstd") == 0);

  struct damacy_config cfg = mk_cfg(root, 1);
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  const int H = 8, W = 22;
  float out[8 * 22] = { 0 };
  size_t got = 0;
  if (run_one(d, mk_sample(p, 3, 11, 5, 27), out, H * W, &got))
    return 1;
  EXPECT(got == (size_t)(H * W));
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      EXPECT(out[y * W + x] == expected_f32_from_u16_2d(3 + y, 5 + x, 32, 0));

  damacy_destroy(d);
  fixture_rm_tree(root);
  return 0;
}

// Four zarrs (none, zstd, blosc-zstd × 2), one sample each in a
// batch_size=4 batch. All three supported codecs route through the
// unified blosc1 GPU pipeline; the wave dispatches to memcpy + nvcomp_zstd.
static int
test_three_codecs_mixed_batch(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  const char* names[4] = { "n", "z", "bz1", "bz2" };
  const char* codecs[4] = { "none", "zstd", "blosc-zstd", "blosc-zstd" };
  const int64_t offsets[4] = { 0, 1000, 2000, 3000 };
  int64_t shape[2] = { 16, 32 }, inner[2] = { 8, 16 }, shard[2] = { 16, 32 };
  char paths[4][256];
  for (int i = 0; i < 4; ++i) {
    snprintf(paths[i], sizeof paths[i], "%s/%s", root, names[i]);
    EXPECT(
      fixture_write_zarr_codec(
        paths[i], shape, inner, shard, 2, "uint16", offsets[i], codecs[i]) ==
      0);
  }

  struct damacy_config cfg = mk_cfg(root, 4);
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  struct damacy_sample s[4];
  for (int i = 0; i < 4; ++i)
    s[i] = mk_sample(paths[i], 0, 16, 0, 32);
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

  // Sub-stage metric sanity: stream_decode must have fired.
  struct damacy_stats st;
  damacy_stats_get(d, &st);
  EXPECT(st.decode.count > 0);

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
  const char* codecs[4] = { "blosc-zstd", "blosc-zstd", "zstd", "blosc-zstd" };
  const int64_t offsets[4] = { 0, 1000, 2000, 3000 };
  int64_t shape[2] = { 16, 32 }, inner[2] = { 8, 16 }, shard[2] = { 16, 32 };
  char paths[4][256];
  for (int i = 0; i < 4; ++i) {
    snprintf(paths[i], sizeof paths[i], "%s/%s", root, names[i]);
    EXPECT(
      fixture_write_zarr_codec(
        paths[i], shape, inner, shard, 2, "uint16", offsets[i], codecs[i]) ==
      0);
  }

  // Tight max_gpu_memory_bytes + small chunk cap drives the resolver
  // to pick the minimum per-wave geometry (one chunk per wave). With
  // max_chunk_uncompressed_bytes = 4 KiB and a budget just barely big
  // enough to fit total_min, dev_decompressed_per_wave lands near
  // 4 KiB. peel_wave's per-chunk read_op is page-aligned (typically
  // 4 KiB), so the 16-chunk batch spills into ≥2 waves of the same
  // batch slot.
  struct damacy_config cfg = {
    .batch_size = 4,
    .lookahead_batches = 2,
    .dtype = DAMACY_F32,
    .device = -1,
    .tuning = {
      .n_io_threads = 1,
      .n_zarrs_meta_cache = 4,
      .n_shards_meta_cache = 4,
      .max_chunk_uncompressed_bytes = 4ull << 10,
      // Resolver minimum so the 16-chunk batch spills into ≥2 waves.
      .max_gpu_memory_bytes = 116ull << 20,
    },
  };
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  struct damacy_sample s[4];
  for (int i = 0; i < 4; ++i)
    s[i] = mk_sample(paths[i], 0, 16, 0, 32);
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

// Single wave with > 32 chunks forces both the per-wave fanout SOA and
// the pool-shared decoder scratch to grow past
// DAMACY_BLOSC_ZSTD_INITIAL_BATCH_CAP (1024). need_zsubs = n_chunks *
// MAX_BLOCKS_PER_CHUNK = 64 * 32 = 2048 > 1024 triggers both grows in
// kick_h2d. Regression for the per-wave-fanout corruption bug: in the
// pre-fix code, growing wave A's fanout reallocated wave B's fanout in
// lockstep — corrupting any in-flight H2D of B's SOA. The new
// per-wave-allocation design prevents this.
//
// 256×32 uint16 zarr with 8×16 inner chunks => 32×2 = 64 chunks per
// sample. Two batches are pushed back-to-back so both waves of the
// pool are kept in flight simultaneously (lookahead_batches=2,
// batch_size=1). Each wave independently triggers its own fanout
// grow + the shared decoder grow on first dispatch.
static int
test_wave_grows_substream_cap(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);
  int64_t shape[2] = { 256, 32 }, inner[2] = { 8, 16 }, shard[2] = { 256, 32 };
  EXPECT(fixture_write_zarr_codec(
           p, shape, inner, shard, 2, "uint16", 0, "blosc-zstd") == 0);

  // Default-budget per-wave geometry easily holds 64 chunks (256 B
  // decompressed each).
  struct damacy_config cfg = {
    .batch_size = 1,
    .lookahead_batches = 2,
    .dtype = DAMACY_F32,
    .device = -1,
    .tuning = {
      .n_io_threads = 1,
      .n_zarrs_meta_cache = 4,
      .n_shards_meta_cache = 4,
    },
  };
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  // Push two samples in one slice so the lookahead can keep both
  // batches in flight; this is what gets the second wave into
  // WAVE_H2D/WAVE_ASSEMBLE concurrently with the first wave's grow.
  struct damacy_sample s[2];
  for (int i = 0; i < 2; ++i)
    s[i] = mk_sample(p, 0, 256, 0, 32);
  struct damacy_sample_slice slice = { .beg = s, .end = s + 2 };
  struct damacy_push_result pr = damacy_push(d, slice);
  EXPECT(pr.status == DAMACY_OK);

  // Pop both batches and verify content.
  for (int iter = 0; iter < 2; ++iter) {
    struct damacy_batch* b = NULL;
    EXPECT(damacy_pop(d, &b) == DAMACY_OK);
    struct damacy_batch_info info;
    damacy_batch_info(b, &info);
    EXPECT(info.shape[0] == 1);
    EXPECT(info.shape[1] == 256);
    EXPECT(info.shape[2] == 32);
    float* out = (float*)calloc(256 * 32, sizeof(float));
    EXPECT(out);
    EXPECT(cudaMemcpy(out,
                      info.device_ptr,
                      256 * 32 * sizeof(float),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int y = 0; y < 256; ++y)
      for (int x = 0; x < 32; ++x)
        EXPECT(out[y * 32 + x] == expected_f32_from_u16_2d(y, x, 32, 0));
    free(out);
    damacy_release(d, b);
  }

  // 2 batches × 64 chunks each.
  struct damacy_stats st;
  damacy_stats_get(d, &st);
  EXPECT(st.batches_emitted == 2);
  EXPECT(st.chunks_dispatched == 2 * 64);

  damacy_destroy(d);
  fixture_rm_tree(root);
  return 0;
}

// Test-only hook from src/damacy.c. See its docstring there.
extern uint64_t
damacy_set_gpu_bytes_committed_for_test(struct damacy* d, uint64_t v);

// The resolver pre-reserves the worst-case grow footprint at
// damacy_create, so the observe-and-grow paths never trip the budget
// when run inside a successfully-created instance. Replays the
// substream-grow workload at a tight budget (resolved per-wave near
// the floor) and asserts the run completes without DAMACY_OOM.
static int
test_grow_inside_tight_budget(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);
  int64_t shape[2] = { 256, 32 }, inner[2] = { 8, 16 }, shard[2] = { 256, 32 };
  EXPECT(fixture_write_zarr_codec(
           p, shape, inner, shard, 2, "uint16", 0, "blosc-zstd") == 0);

  struct damacy_config cfg = {
    .batch_size = 1,
    .lookahead_batches = 2,
    .dtype = DAMACY_F32,
    .device = -1,
    .tuning = {
      .n_io_threads = 1,
      .n_zarrs_meta_cache = 4,
      .n_shards_meta_cache = 4,
      .max_chunk_uncompressed_bytes = 4ull << 10,
      .max_gpu_memory_bytes = 120ull << 20,
    },
  };
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  struct damacy_sample s = mk_sample(p, 0, 256, 0, 32);
  struct damacy_sample_slice slice = { .beg = &s, .end = &s + 1 };
  EXPECT(damacy_push(d, slice).status == DAMACY_OK);

  // 64 chunks × 32 blocks/chunk = 2048 substreams, > the 1024 initial
  // floor, so both fanout + decoder grows fire inside the tight budget.
  // The grow paths must not return DAMACY_OOM here — the resolver
  // reserved the worst-case footprint.
  struct damacy_batch* b = NULL;
  EXPECT(damacy_pop(d, &b) == DAMACY_OK);
  damacy_release(d, b);

  damacy_destroy(d);
  fixture_rm_tree(root);
  return 0;
}

// With the planner's chunk_layout probe in place, the per-wave
// substream count is computed from the probed nblocks rather than
// MAX_BLOCKS_PER_CHUNK. This 256-byte-chunk fixture has nblocks=1, so
// need_zsubs (64) stays under the initial 1024-cap and the
// decoder/fanout grow paths do NOT fire — even with the budget
// inflated to leave only 1 MB headroom. The pop succeeds.
//
// The OOM-on-grow code path itself is unchanged; it just isn't
// reachable for this fixture under the tight bound. A workload that
// reaches it would need multi-block chunks (nblocks > 1 from blosc's
// auto-tune, i.e., chunks much larger than 16 KB) and enough chunks
// per wave to push past 1024 substreams.
static int
test_layout_probe_avoids_decoder_grow(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);
  int64_t shape[2] = { 256, 32 }, inner[2] = { 8, 16 }, shard[2] = { 256, 32 };
  EXPECT(fixture_write_zarr_codec(
           p, shape, inner, shard, 2, "uint16", 0, "blosc-zstd") == 0);

  struct damacy_config cfg = {
    .batch_size = 1,
    .lookahead_batches = 2,
    .dtype = DAMACY_F32,
    .device = -1,
    .tuning = {
      .n_io_threads = 1,
      .n_zarrs_meta_cache = 4,
      .n_shards_meta_cache = 4,
      .max_chunk_uncompressed_bytes = 4ull << 10,
      .max_gpu_memory_bytes = 120ull << 20,
    },
  };
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  // Pre-fix: the decoder grow's ~7 MB delta from 1024 → 2048
  // substreams (driven by need_zsubs = n_chunks * MAX_BLOCKS) tripped
  // OOM here. Post-fix: probed nblocks = 1, need_zsubs = 64, no grow.
  const uint64_t headroom = 1ull << 20;
  (void)damacy_set_gpu_bytes_committed_for_test(
    d, cfg.tuning.max_gpu_memory_bytes - headroom);

  struct damacy_sample s = mk_sample(p, 0, 256, 0, 32);
  struct damacy_sample_slice slice = { .beg = &s, .end = &s + 1 };
  EXPECT(damacy_push(d, slice).status == DAMACY_OK);

  struct damacy_batch* b = NULL;
  EXPECT(damacy_pop(d, &b) == DAMACY_OK);
  damacy_release(d, b);

  damacy_destroy(d);
  fixture_rm_tree(root);
  return 0;
}

// Exercise the batch-output pool's OOM branch with the
// gpu_bytes_committed counter inflated to the cap. Belt-and-suspenders
// alongside test_layout_probe_avoids_decoder_grow: this one shows that
// the same hook plus a much tighter headroom trips the earlier
// batch_pool_allocate check, so neither grow path is reached.
static int
test_batch_pool_rejected_at_inflated_committed(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);
  int64_t shape[2] = { 8, 16 }, inner[2] = { 4, 8 }, shard[2] = { 8, 16 };
  EXPECT(fixture_write_zarr_codec(
           p, shape, inner, shard, 2, "uint16", 0, "blosc-zstd") == 0);

  struct damacy_config cfg = mk_cfg(root, 1);
  cfg.tuning.max_gpu_memory_bytes = 120ull << 20;
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  // No headroom at all — any byte added is over the cap.
  (void)damacy_set_gpu_bytes_committed_for_test(
    d, cfg.tuning.max_gpu_memory_bytes);

  struct damacy_sample s = mk_sample(p, 0, 8, 0, 16);
  struct damacy_sample_slice slice = { .beg = &s, .end = &s + 1 };
  EXPECT(damacy_push(d, slice).status == DAMACY_OK);

  struct damacy_batch* b = NULL;
  EXPECT(damacy_pop(d, &b) == DAMACY_OOM);

  damacy_destroy(d);
  fixture_rm_tree(root);
  return 0;
}

// Returns 1 iff store_fs_gds_create succeeds on this host (libcufile
// loadable, driver opens). 0 when GDS isn't built or isn't available.
static int
gds_runtime_available(void)
{
  char tmpl[] = "/tmp/damacy_gds_probe_XXXXXX";
  char* root = mkdtemp(tmpl);
  if (!root)
    return 0;
  struct store_fs_config sc = { .root = root, .nthreads = 1 };
  struct store* s = store_fs_gds_create(&sc);
  int ok = (s != NULL);
  store_destroy(s);
  fixture_rm_tree(root);
  return ok;
}

// Host-staging vs GDS through the GPU parse pipeline. Skipped (logged +
// returns 0) when cuFile is unavailable on this host.
static int
test_gds_parity_blosc_zstd(void)
{
  if (!gds_runtime_available()) {
    log_info("test_gds_parity_blosc_zstd: GDS unavailable on this host; skip");
    return 0;
  }
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);
  int64_t shape[2] = { 64, 32 };
  int64_t inner[2] = { 8, 16 };
  int64_t shard[2] = { 64, 32 };
  EXPECT(fixture_write_zarr_codec(
           p, shape, inner, shard, 2, "uint16", 0, "blosc-zstd") == 0);

  const size_t n_elems = (size_t)shape[0] * (size_t)shape[1];
  float* host_out = (float*)calloc(n_elems, sizeof(float));
  float* gds_out = (float*)calloc(n_elems, sizeof(float));
  EXPECT(host_out && gds_out);

  // Host-staging path.
  {
    struct damacy_config cfg = mk_cfg(root, 1);
    struct damacy* d = NULL;
    EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);
    size_t got = 0;
    EXPECT(run_one(d, mk_sample(p, 0, 64, 0, 32), host_out, n_elems, &got) ==
           0);
    EXPECT(got == n_elems);
    damacy_destroy(d);
  }
  // GDS path.
  {
    struct damacy_config cfg = mk_cfg(root, 1);
    cfg.tuning.enable_gds = 1;
    struct damacy* d = NULL;
    EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);
    size_t got = 0;
    EXPECT(run_one(d, mk_sample(p, 0, 64, 0, 32), gds_out, n_elems, &got) == 0);
    EXPECT(got == n_elems);
    damacy_destroy(d);
  }

  for (size_t i = 0; i < n_elems; ++i)
    EXPECT(host_out[i] == gds_out[i]);

  free(host_out);
  free(gds_out);
  fixture_rm_tree(root);
  return 0;
}

int
main(void)
{
  EXPECT(cuda_init_primary() == 0);
  RUN(test_full_array_blosc_zstd);
  RUN(test_partial_crossing_chunks_blosc);
  RUN(test_three_codecs_mixed_batch);
  RUN(test_multi_wave_per_batch);
  RUN(test_wave_grows_substream_cap);
  RUN(test_grow_inside_tight_budget);
  RUN(test_layout_probe_avoids_decoder_grow);
  RUN(test_batch_pool_rejected_at_inflated_committed);
  RUN(test_gds_parity_blosc_zstd);
  log_info("all tests passed");
  return 0;
}
