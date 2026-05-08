// Runtime-cap tests: damacy_config.max_chunk_uncompressed_bytes and
// max_gpu_memory_bytes (issue #7).
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

#include "cuda_init.h"
#include "damacy.h"
#include "decoder/decoder_lz4.h"
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
    .device = -1,
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
           p, shape, inner, shard, 2, "uint16", 0, "blosc-lz4") == 0);

  struct damacy_config cfg = mk_cfg(root, 1);
  // Both runtime knobs at 0 — defaults: 512 KB chunk cap, no GPU cap.
  cfg.max_chunk_uncompressed_bytes = 0;
  cfg.max_gpu_memory_bytes = 0;

  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  struct damacy_sample s = mk_sample("foo", 0, 8, 0, 16);
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
           p, shape, inner, shard, 2, "uint16", 0, "blosc-lz4") == 0);

  struct damacy_config cfg = mk_cfg(root, 1);
  cfg.max_chunk_uncompressed_bytes = 128; // 8x16 u16 = 256 B → over

  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  struct damacy_sample s = mk_sample("foo", 0, 8, 0, 16);
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
  struct damacy_config cfg = mk_cfg(root, 1);
  cfg.max_chunk_uncompressed_bytes =
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
  struct damacy_config cfg = mk_cfg(root, 1);
  cfg.max_gpu_memory_bytes = 64;
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

  EXPECT(decoder_lz4_query_temp_bytes(batch, 4096, total, &small) == 0);
  EXPECT(decoder_lz4_query_temp_bytes(batch, 512ull << 10, total, &large) == 0);
  EXPECT(small <= large);

  cuDevicePrimaryCtxRelease(dev);
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
  log_info("all tests passed");
  return 0;
}
