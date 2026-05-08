// Regression tests for issue #12: damacy_create's CUDA context
// contract. damacy_create captures the caller's current CUcontext;
// pre-fix it stomped dev 0's primary, segfaulting DDP on the next
// DLPack import.
//
// Test cases:
//   test_create_preserves_caller_ctx — caller has a non-primary ctx
//                                      current; create + destroy
//                                      must not change it.
//   test_create_no_ctx_returns_inval — no CUcontext current; create
//                                      must return DAMACY_INVAL.

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
mk_cfg(const char* root)
{
  return (struct damacy_config){
    .store_root = root,
    .batch_size = 1,
    .lookahead_batches = 2,
    .n_io_threads = 1,
    .host_buffer_bytes = 1ull << 20,
    .device_buffer_bytes = 1ull << 20,
    .n_zarrs_meta_cache = 4,
    .n_shards_meta_cache = 4,
    .dtype = DAMACY_U16,
  };
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
  // an identity change in cuCtxGetCurrent.
  CUcontext caller = NULL;
  EXPECT(cuCtxCreate(&caller, NULL, 0, dev) == CUDA_SUCCESS);
  EXPECT(cuCtxSetCurrent(caller) == CUDA_SUCCESS);

  CUcontext before = NULL;
  EXPECT(cuCtxGetCurrent(&before) == CUDA_SUCCESS);
  EXPECT(before == caller);

  char root[64];
  EXPECT(mkdtemp_root(root, sizeof root) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);
  int64_t shape[2] = { 8, 16 }, inner[2] = { 4, 8 }, shard[2] = { 8, 16 };
  EXPECT(fixture_write_zarr_codec(
           p, shape, inner, shard, 2, "uint16", 0, "blosc-lz4") == 0);

  struct damacy_config cfg = mk_cfg(root);
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
  struct damacy_config cfg = mk_cfg(root);
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_INVAL);
  EXPECT(d == NULL);
  fixture_rm_tree(root);
  return 0;
}

int
main(void)
{
  RUN(test_create_preserves_caller_ctx);
  RUN(test_create_no_ctx_returns_inval);
  log_info("all tests passed");
  return 0;
}
