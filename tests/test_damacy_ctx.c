// Regression tests for damacy's per-call CUDA context contract.
//
// Test cases:
//   test_create_preserves_caller_ctx
//                            — non-primary caller ctx survives create+destroy
//   test_create_no_ctx_returns_inval
//                            — no ctx current and cfg.device<0 -> INVAL
//   test_create_explicit_device_no_ctx
//                            — cfg.device set, no caller ctx -> retain primary
//   test_create_explicit_device_mismatch
//                            — cfg.device != current ctx's device -> INVAL
//   test_explicit_device_does_not_leak_ctx_between_calls
//                            — push/pop/release/flush/destroy restore caller
//   test_pop_error_path_restores_caller_ctx
//                            — pop failing inside the guard still pops ctx

#include "damacy.h"
#include "fixture.h"

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int
mkdtemp_root(char* root, size_t cap)
{
  if (cap < sizeof "/tmp/damacy_ctx_XXXXXX")
    return 1;
  strcpy(root, "/tmp/damacy_ctx_XXXXXX");
  return mkdtemp(root) ? 0 : 1;
}

static struct damacy_config
mk_cfg(const char* root, int64_t sy, int64_t sx)
{
  (void)root;
  struct damacy_config c = {
    .samples_per_batch = 1,
    .lookahead_samples = 2,
    .dtype = DAMACY_F32,
    .sample_rank = 2,
    .device = -1,
    .tuning = {
      .n_io_threads = 1,
      .n_prefetch_threads = 1,
      .n_metadata_io_threads = 1,
      .n_array_meta_cache = 4,
      .n_shard_index_cache = 4,
      .n_chunk_layout_cache = 4,
      .max_gpu_memory_bytes = 1ull << 30,
    },
  };
  c.sample_shape[0] = sy;
  c.sample_shape[1] = sx;
  return c;
}

// Caller has a non-primary CUcontext current. damacy_create captures
// it and damacy_destroy leaves it current — pre-fix, dev 0's primary
// would have replaced it (observable as a CUcontext identity change).
static int
test_create_preserves_caller_ctx(void)
{
  EXPECT(cuInit(0) == CUDA_SUCCESS);
  CUdevice dev = 0;
  EXPECT(cuDeviceGet(&dev, 0) == CUDA_SUCCESS);

  // Non-primary ctx with default params — distinct identity from the
  // primary, so any "set primary current" regression is observable as
  // an identity change in cuCtxGetCurrent. cuCtxCreate already makes
  // the new ctx current.
  CUcontext caller = NULL;
  EXPECT(cuCtxCreate(&caller, NULL, 0, dev) == CUDA_SUCCESS);

  CUcontext before = NULL;
  EXPECT(cuCtxGetCurrent(&before) == CUDA_SUCCESS);
  EXPECT(before == caller);

  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);
  int64_t shape[2] = { 8, 16 }, inner[2] = { 4, 8 }, shard[2] = { 8, 16 };
  EXPECT(fixture_write_zarr_codec(
           p, shape, inner, shard, 2, "uint16", 0, "blosc-zstd") == 0);

  struct damacy_config cfg = mk_cfg(root, 8, 16);
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  CUcontext after_create = NULL;
  EXPECT(cuCtxGetCurrent(&after_create) == CUDA_SUCCESS);
  EXPECT(after_create == caller);

  damacy_destroy(d);

  CUcontext after_destroy = NULL;
  EXPECT(cuCtxGetCurrent(&after_destroy) == CUDA_SUCCESS);
  EXPECT(after_destroy == caller);

  EXPECT(cuCtxSetCurrent(NULL) == CUDA_SUCCESS);
  EXPECT(cuCtxDestroy(caller) == CUDA_SUCCESS);
  fixture_rm_tree(root);
  return 0;
}

// No CUcontext current — damacy_create must reject with INVAL.
static int
test_create_no_ctx_returns_inval(void)
{
  EXPECT(cuInit(0) == CUDA_SUCCESS);
  EXPECT(cuCtxSetCurrent(NULL) == CUDA_SUCCESS);

  CUcontext cur = (CUcontext)0x1; // sentinel
  EXPECT(cuCtxGetCurrent(&cur) == CUDA_SUCCESS);
  EXPECT(cur == NULL);

  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  struct damacy_config cfg = mk_cfg(root, 1, 1);
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_INVAL);
  EXPECT(d == NULL);
  fixture_rm_tree(root);
  return 0;
}

// cfg.device set with no caller context: damacy retains primary itself.
static int
test_create_explicit_device_no_ctx(void)
{
  EXPECT(cuInit(0) == CUDA_SUCCESS);
  EXPECT(cuCtxSetCurrent(NULL) == CUDA_SUCCESS);

  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);
  int64_t shape[2] = { 8, 16 }, inner[2] = { 4, 8 }, shard[2] = { 8, 16 };
  EXPECT(fixture_write_zarr_codec(
           p, shape, inner, shard, 2, "uint16", 0, "blosc-zstd") == 0);

  struct damacy_config cfg = mk_cfg(root, 8, 16);
  cfg.device = 0;
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);
  EXPECT(damacy_get_device(d) == 0);
  damacy_destroy(d);
  fixture_rm_tree(root);
  return 0;
}

// cfg.device set, but a context is already current on a different
// device. Mismatch is rejected with INVAL.
static int
test_create_explicit_device_mismatch(void)
{
  int n_devices = 0;
  EXPECT(cuInit(0) == CUDA_SUCCESS);
  EXPECT(cuDeviceGetCount(&n_devices) == CUDA_SUCCESS);
  if (n_devices < 2) {
    log_info("skip: needs >= 2 CUDA devices (have %d)", n_devices);
    return 0;
  }
  CUdevice dev0 = 0;
  EXPECT(cuDeviceGet(&dev0, 0) == CUDA_SUCCESS);
  CUcontext caller = NULL;
  EXPECT(cuCtxCreate(&caller, NULL, 0, dev0) == CUDA_SUCCESS);

  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  struct damacy_config cfg = mk_cfg(root, 1, 1);
  cfg.device = 1;
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_INVAL);
  EXPECT(d == NULL);

  EXPECT(cuCtxSetCurrent(NULL) == CUDA_SUCCESS);
  EXPECT(cuCtxDestroy(caller) == CUDA_SUCCESS);
  fixture_rm_tree(root);
  return 0;
}

static int
test_explicit_device_does_not_leak_ctx_between_calls(void)
{
  EXPECT(cuInit(0) == CUDA_SUCCESS);
  CUdevice dev = 0;
  EXPECT(cuDeviceGet(&dev, 0) == CUDA_SUCCESS);

  CUcontext caller = NULL;
  EXPECT(cuCtxCreate(&caller, NULL, 0, dev) == CUDA_SUCCESS);

  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);
  int64_t shape[2] = { 4, 8 }, inner[2] = { 2, 4 }, shard[2] = { 4, 8 };
  EXPECT(fixture_write_zarr_codec(
           p, shape, inner, shard, 2, "uint16", 0, "blosc-zstd") == 0);

  struct damacy_config cfg = mk_cfg(root, 4, 8);
  cfg.device = 0;
  cfg.samples_per_batch = 1;
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  CUcontext cur = NULL;
  EXPECT(cuCtxGetCurrent(&cur) == CUDA_SUCCESS);
  EXPECT(cur == caller);

  struct damacy_sample s = { .uri = p, .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 4 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 8 };
  struct damacy_sample_slice slice = { .beg = &s, .end = &s + 1 };
  struct damacy_push_result pr = damacy_push(d, slice);
  EXPECT(pr.status == DAMACY_OK);
  EXPECT(cuCtxGetCurrent(&cur) == CUDA_SUCCESS);
  EXPECT(cur == caller);

  struct damacy_batch* b = NULL;
  EXPECT(damacy_pop(d, &b) == DAMACY_OK);
  EXPECT(cuCtxGetCurrent(&cur) == CUDA_SUCCESS);
  EXPECT(cur == caller);
  damacy_release(d, b);

  EXPECT(damacy_flush(d) == DAMACY_OK);
  EXPECT(cuCtxGetCurrent(&cur) == CUDA_SUCCESS);
  EXPECT(cur == caller);

  damacy_destroy(d);
  EXPECT(cuCtxGetCurrent(&cur) == CUDA_SUCCESS);
  EXPECT(cur == caller);

  EXPECT(cuCtxSetCurrent(NULL) == CUDA_SUCCESS);
  EXPECT(cuCtxDestroy(caller) == CUDA_SUCCESS);
  fixture_rm_tree(root);
  return 0;
}

// damacy_pop fails after ctx_guard_enter has succeeded: an out-of-bounds
// AABB is accepted by push (which doesn't validate against meta shape)
// but rejected by the planner inside kick_new_waves. Verify the guard
// still pops the caller ctx on the failure path.
static int
test_pop_error_path_restores_caller_ctx(void)
{
  EXPECT(cuInit(0) == CUDA_SUCCESS);
  CUdevice dev = 0;
  EXPECT(cuDeviceGet(&dev, 0) == CUDA_SUCCESS);

  CUcontext caller = NULL;
  EXPECT(cuCtxCreate(&caller, NULL, 0, dev) == CUDA_SUCCESS);

  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);
  int64_t shape[2] = { 4, 8 }, inner[2] = { 2, 4 }, shard[2] = { 4, 8 };
  EXPECT(fixture_write_zarr_codec(
           p, shape, inner, shard, 2, "uint16", 0, "blosc-zstd") == 0);

  // sample_shape matches the OOB AABB extents so push validates the
  // sample through; the planner is the one that rejects the AABB for
  // extending past the zarr's actual shape.
  struct damacy_config cfg = mk_cfg(root, 4, 9999);
  cfg.device = 0;
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  CUcontext cur = NULL;
  EXPECT(cuCtxGetCurrent(&cur) == CUDA_SUCCESS);
  EXPECT(cur == caller);

  // Push succeeds (shape matches cfg), pop's planner rejects the AABB
  // with DAMACY_INVAL after ctx_guard_enter.
  struct damacy_sample oob = { .uri = p, .aabb = { .rank = 2 } };
  oob.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 4 };
  oob.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 9999 };
  struct damacy_sample_slice slice = { .beg = &oob, .end = &oob + 1 };
  struct damacy_push_result pr = damacy_push(d, slice);
  EXPECT(pr.status == DAMACY_OK);

  struct damacy_batch* b = NULL;
  EXPECT(damacy_pop(d, &b) != DAMACY_OK);
  EXPECT(cuCtxGetCurrent(&cur) == CUDA_SUCCESS);
  EXPECT(cur == caller);

  damacy_destroy(d);
  EXPECT(cuCtxGetCurrent(&cur) == CUDA_SUCCESS);
  EXPECT(cur == caller);

  EXPECT(cuCtxSetCurrent(NULL) == CUDA_SUCCESS);
  EXPECT(cuCtxDestroy(caller) == CUDA_SUCCESS);
  fixture_rm_tree(root);
  return 0;
}

int
main(void)
{
  RUN(test_create_preserves_caller_ctx);
  RUN(test_create_no_ctx_returns_inval);
  RUN(test_create_explicit_device_no_ctx);
  RUN(test_create_explicit_device_mismatch);
  RUN(test_explicit_device_does_not_leak_ctx_between_calls);
  RUN(test_pop_error_path_restores_caller_ctx);
  log_info("all tests passed");
  return 0;
}
