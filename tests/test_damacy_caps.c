// Runtime-cap tests: damacy_config.max_chunk_uncompressed_bytes,
// max_gpu_memory_bytes, and the resolver's batch-output pool reserve
// (derived from cfg.sample_shape).
//
// Test cases:
//   test_create_default_caps        — 0/0 config matches pre-existing
//                                     behaviour, instance comes up fine
//   test_oversize_chunk_rejected    — set chunk cap below the inner-chunk
//                                     uncompressed size; planner returns
//                                     DAMACY_INVAL via damacy_pop
//   test_chunk_cap_too_high         — value > compile-time ceiling rejected
//                                     at create with DAMACY_INVAL
//   test_gpu_budget_too_small       — max_gpu_memory_bytes set absurdly
//                                     low; create returns DAMACY_OOM
//   test_chunk_cap_shrinks_nvcomp   — smaller runtime cap → smaller nvcomp
//                                     temp scratch (queried directly)
//   test_config_describe            — damacy_config_describe runs and
//                                     logs without touching CUDA state
//   test_pool_reserve_fits_default_budget
//                                   — non-trivial pool reserve at the
//                                     default cap; create+push+pop OK,
//                                     committed counter stays under cap
//   test_pool_exceeds_budget_rejected_at_create
//                                   — pool reserve > cap; create returns
//                                     DAMACY_OOM
//   test_sample_shape_mismatch_rejected
//                                   — push a sample whose aabb extent !=
//                                     cfg.sample_shape; expect INVAL
//   test_resolver_minimum_one_chunk — budget barely fitting a
//                                     single-chunk wave produces a valid
//                                     instance and surfaces tight geometry

#include "cuda_init.h"
#include "damacy.h"
#include "decoder/decoder_zstd.h"
#include "fixture.h"

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int
mkdtemp_root(char* root, size_t cap)
{
  if (cap < sizeof "/tmp/damacy_caps_XXXXXX")
    return 1;
  strcpy(root, "/tmp/damacy_caps_XXXXXX");
  return mkdtemp(root) ? 0 : 1;
}

static struct damacy_config
mk_cfg(const char* root, uint32_t batch_size, int64_t sy, int64_t sx)
{
  (void)root;
  struct damacy_config c = {
    .batch_size = batch_size,
    .lookahead_batches = 2,
    .dtype = DAMACY_F32,
    .sample_rank = 2,
    .device = -1,
    .tuning = {
      .n_io_threads = 1,
      .n_zarrs_meta_cache = 4,
      .n_shards_meta_cache = 4,
    },
  };
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

// Default config should still bring up the pipeline and run a sample —
// proves the runtime-cap default (512 KB) lines up with the pre-runtime
// behaviour for the existing test workloads.
static int
test_create_default_caps(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);
  int64_t shape[2] = { 8, 16 }, inner[2] = { 4, 8 }, shard[2] = { 8, 16 };
  EXPECT(fixture_write_zarr_codec(
           p, shape, inner, shard, 2, "uint16", 0, "blosc-zstd") == 0);

  struct damacy_config cfg = mk_cfg(root, 1, 8, 16);
  // Both runtime knobs at 0 — defaults: 512 KB chunk cap, no GPU cap.
  cfg.tuning.max_chunk_uncompressed_bytes = 0;
  cfg.tuning.max_gpu_memory_bytes = 0;

  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  struct damacy_sample s = mk_sample(p, 0, 8, 0, 16);
  struct damacy_sample_slice slice = { .beg = &s, .end = &s + 1 };
  EXPECT(damacy_push(d, slice).status == DAMACY_OK);
  struct damacy_batch* b = NULL;
  EXPECT(damacy_pop(d, &b) == DAMACY_OK);
  damacy_release(d, b);

  damacy_destroy(d);
  fixture_rm_tree(root);
  return 0;
}

// Inner chunk = 8x16 u16 = 256 bytes. Set runtime cap = 128 bytes; the
// planner must reject the chunk with DAMACY_INVAL (surfaces through the
// pop after push consumes the sample into the lookahead).
static int
test_oversize_chunk_rejected(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);
  int64_t shape[2] = { 8, 16 }, inner[2] = { 8, 16 }, shard[2] = { 8, 16 };
  EXPECT(fixture_write_zarr_codec(
           p, shape, inner, shard, 2, "uint16", 0, "blosc-zstd") == 0);

  struct damacy_config cfg = mk_cfg(root, 1, 8, 16);
  cfg.tuning.max_chunk_uncompressed_bytes = 128; // 8x16 u16 = 256 B → over

  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  struct damacy_sample s = mk_sample(p, 0, 8, 0, 16);
  struct damacy_sample_slice slice = { .beg = &s, .end = &s + 1 };
  // push validates dtype/rank only; the planner rejection lands at pop.
  EXPECT(damacy_push(d, slice).status == DAMACY_OK);
  struct damacy_batch* b = NULL;
  EXPECT(damacy_pop(d, &b) == DAMACY_INVAL);

  damacy_destroy(d);
  fixture_rm_tree(root);
  return 0;
}

// Values above DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES (the kernel-array
// ceiling) must be rejected at create — no pipeline state has been set
// up at the failure point.
static int
test_chunk_cap_too_high(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  struct damacy_config cfg = mk_cfg(root, 1, 8, 16);
  cfg.tuning.max_chunk_uncompressed_bytes =
    (uint32_t)DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES + 1u;
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_INVAL);
  EXPECT(d == NULL);
  fixture_rm_tree(root);
  return 0;
}

// max_gpu_memory_bytes set absurdly low (64 B) — wave-resident memory
// at default config is many MB, so create must fail with DAMACY_OOM.
static int
test_gpu_budget_too_small(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  struct damacy_config cfg = mk_cfg(root, 1, 8, 16);
  cfg.tuning.max_gpu_memory_bytes = 64;
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OOM);
  EXPECT(d == NULL);
  fixture_rm_tree(root);
  return 0;
}

// Smaller runtime per-substream cap should produce a strictly smaller
// (or equal) nvcomp temp scratch. We probe the same query helpers that
// damacy_create + wave_init use to size scratch.
static int
test_chunk_cap_shrinks_nvcomp(void)
{
  CUcontext ctx = NULL;
  EXPECT(cuInit(0) == CUDA_SUCCESS);
  CUdevice dev = 0;
  EXPECT(cuDeviceGet(&dev, 0) == CUDA_SUCCESS);
  EXPECT(cuDevicePrimaryCtxRetain(&ctx, dev) == CUDA_SUCCESS);
  EXPECT(cuCtxSetCurrent(ctx) == CUDA_SUCCESS);

  const size_t batch = 256;
  const size_t total = 8ull << 20;

  size_t small = 0, large = 0;
  EXPECT(decoder_zstd_query_temp_bytes(batch, 4096, total, &small) == 0);
  EXPECT(decoder_zstd_query_temp_bytes(batch, 512ull << 10, total, &large) ==
         0);
  // nvcomp's scratch grows with per-substream cap; smaller cap mustn't
  // ask for more memory than the larger one.
  EXPECT(small <= large);

  cuDevicePrimaryCtxRelease(dev);
  return 0;
}

// damacy_config_describe should run safely on a sane config without
// constructing a damacy instance. It does call into nvcomp via
// decoder_zstd_query_temp_bytes, so a live CUDA context is required —
// cuda_init_primary in main() retains the primary on dev 0 before any
// describe call. Output goes through log/log.h — no return value to
// check; we just want it not to crash.
static int
test_config_describe(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  struct damacy_config cfg = mk_cfg(root, 1, 8, 16);
  cfg.tuning.max_gpu_memory_bytes = 0;
  damacy_config_describe(&cfg);
  cfg.tuning.max_gpu_memory_bytes = 256ull << 20;
  damacy_config_describe(&cfg);
  // NULL-safe sanity: must not crash.
  damacy_config_describe(NULL);
  fixture_rm_tree(root);
  return 0;
}

// Issue #59 regime: a non-trivial batch-output pool (sample volume ×
// batch_size × dtype_bpe × 2) at the default cap. The resolver carves
// out pool_reserve from max_gpu_memory_bytes before sizing the wave-
// resident buffers, so create + push + pop succeed and the committed
// counter stays at or below the cap after the pop. Pre-fix, the lazy
// batch_pool_allocate at first push tripped OOM here.
static int
test_pool_reserve_fits_default_budget(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);
  // Rank-4 sample matching the issue's reproducer geometry.
  // Inner chunks kept ≤ default chunk cap (512 KB): 4×1×64×64 u16
  // = 32 KB per chunk; 64 chunks per shard.
  int64_t shape[4] = { 16, 1, 256, 256 };
  int64_t inner[4] = { 4, 1, 64, 64 };
  int64_t shard[4] = { 16, 1, 256, 256 };
  EXPECT(fixture_write_zarr_codec(
           p, shape, inner, shard, 4, "uint16", 0, "blosc-zstd") == 0);

  // batch_size=20, sample=(16,1,256,256) f32 → 80 MB per slot, 160 MB
  // double-buffered. Default cap is 1 GiB; ~860 MB left for the
  // resolver, comfortably fits the wave-resident geometry.
  struct damacy_config cfg = {
    .batch_size = 20,
    .lookahead_batches = 2,
    .dtype = DAMACY_F32,
    .sample_shape = { 16, 1, 256, 256 },
    .sample_rank = 4,
    .device = -1,
    .tuning = {
      .n_io_threads = 1,
      .n_zarrs_meta_cache = 4,
      .n_shards_meta_cache = 4,
    },
  };
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  struct damacy_sample samples[20];
  for (int i = 0; i < 20; ++i) {
    samples[i].uri = p;
    samples[i].aabb.rank = 4;
    samples[i].aabb.dims[0] = (struct damacy_interval){ 0, 16 };
    samples[i].aabb.dims[1] = (struct damacy_interval){ 0, 1 };
    samples[i].aabb.dims[2] = (struct damacy_interval){ 0, 256 };
    samples[i].aabb.dims[3] = (struct damacy_interval){ 0, 256 };
  }
  struct damacy_sample_slice slice = { .beg = samples, .end = samples + 20 };
  EXPECT(damacy_push(d, slice).status == DAMACY_OK);

  struct damacy_batch* b = NULL;
  EXPECT(damacy_pop(d, &b) == DAMACY_OK);
  damacy_release(d, b);

  struct damacy_stats st;
  damacy_stats_get(d, &st);
  EXPECT(st.gpu_bytes_committed > 0);
  // Default cap when cfg.max_gpu_memory_bytes is left at 0.
  EXPECT(st.gpu_bytes_committed <= DAMACY_DEFAULT_MAX_GPU_MEMORY_BYTES);

  damacy_destroy(d);
  fixture_rm_tree(root);
  return 0;
}

// Same sample geometry as test_pool_reserve_fits_default_budget but
// cfg.max_gpu_memory_bytes set below the required pool_reserve. The
// resolver rejects at damacy_create with DAMACY_OOM — nothing is left
// for wave-resident buffers.
static int
test_pool_exceeds_budget_rejected_at_create(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  struct damacy_config cfg = {
    .batch_size = 20,
    .lookahead_batches = 2,
    .dtype = DAMACY_F32,
    .sample_shape = { 16, 1, 256, 256 },
    .sample_rank = 4,
    .device = -1,
    .tuning = {
      .n_io_threads = 1,
      .n_zarrs_meta_cache = 4,
      .n_shards_meta_cache = 4,
      // pool_reserve = 2 × 20 × 16 × 256 × 256 × 4 ≈ 160 MB, well above
      // the 32 MB cap below — create fails before sizing.
      .max_gpu_memory_bytes = 32ull << 20,
    },
  };
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OOM);
  EXPECT(d == NULL);
  fixture_rm_tree(root);
  return 0;
}

// push validates each sample's aabb extents against cfg->sample_shape.
// A sample with a different extent is rejected with DAMACY_INVAL at
// push time, before the lookahead absorbs it.
static int
test_sample_shape_mismatch_rejected(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);
  int64_t shape[2] = { 8, 16 }, inner[2] = { 4, 8 }, shard[2] = { 8, 16 };
  EXPECT(fixture_write_zarr_codec(
           p, shape, inner, shard, 2, "uint16", 0, "blosc-zstd") == 0);

  // cfg.sample_shape = (8, 16); push a sample with extent (4, 8).
  struct damacy_config cfg = mk_cfg(root, 1, 8, 16);
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  struct damacy_sample s = mk_sample(p, 0, 4, 0, 8);
  struct damacy_sample_slice slice = { .beg = &s, .end = &s + 1 };
  struct damacy_push_result pr = damacy_push(d, slice);
  EXPECT(pr.status == DAMACY_INVAL);
  // Offending sample is at the head of the unconsumed suffix.
  EXPECT(pr.unconsumed.beg == &s);

  damacy_destroy(d);
  fixture_rm_tree(root);
  return 0;
}

// Resolver picks per-wave near the minimum when the budget barely
// fits. Reports gpu_bytes_committed back through stats so users can
// observe the resolved size.
static int
test_resolver_minimum_one_chunk(void)
{
  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);
  int64_t shape[2] = { 8, 16 }, inner[2] = { 4, 8 }, shard[2] = { 8, 16 };
  EXPECT(fixture_write_zarr_codec(
           p, shape, inner, shard, 2, "uint16", 0, "blosc-zstd") == 0);

  struct damacy_config cfg = mk_cfg(root, 1, 8, 16);
  // Very small per-chunk cap + budget close to the resolved minimum.
  cfg.tuning.max_chunk_uncompressed_bytes = 4ull << 10;
  cfg.tuning.max_gpu_memory_bytes = 120ull << 20;

  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  struct damacy_stats st;
  damacy_stats_get(d, &st);
  // gpu_bytes_committed reflects the initial alloc and must be > 0
  // (waves are up). It's <= the configured cap by construction.
  EXPECT(st.gpu_bytes_committed > 0);
  EXPECT(st.gpu_bytes_committed <= cfg.tuning.max_gpu_memory_bytes);

  damacy_destroy(d);
  fixture_rm_tree(root);
  return 0;
}

int
main(void)
{
  EXPECT(cuda_init_primary() == 0);
  RUN(test_create_default_caps);
  RUN(test_oversize_chunk_rejected);
  RUN(test_chunk_cap_too_high);
  RUN(test_gpu_budget_too_small);
  RUN(test_chunk_cap_shrinks_nvcomp);
  RUN(test_config_describe);
  RUN(test_pool_reserve_fits_default_budget);
  RUN(test_pool_exceeds_budget_rejected_at_create);
  RUN(test_sample_shape_mismatch_rejected);
  RUN(test_resolver_minimum_one_chunk);
  log_info("all tests passed");
  return 0;
}
