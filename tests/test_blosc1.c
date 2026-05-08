// End-to-end coverage of the host-side blosc1 parse: walks the six
// fixture chunks in experiments/blosc1-gpu, runs blosc1_host_parse, then
// drives nvcomp + (bit)unshuffle on the resulting fanout / op arrays
// and memcmp's the decompressed output against the .raw ground truth.
//
// Bit-exact equivalence is the contract: any bug in the host parse,
// emit, or rank-sort surfaces here as either a non-zero h_hdr.err, a
// failed flag-extraction assert, or a bad memcmp.

#include "decoder/bitshuffle.h"
#include "decoder/blosc1.h"
#include "decoder/blosc1_host.h"
#include "decoder/decoder_lz4.h"
#include "decoder/decoder_memcpy.h"
#include "decoder/decoder_zstd.h"
#include "decoder/shuffle.h"
#include "expect.h"
#include "threadpool/threadpool.h"
#include "zarr/zarr_metadata.h"

#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifndef BLOSC1_FIXTURES_DIR
#error "BLOSC1_FIXTURES_DIR must be defined by the build system"
#endif
#ifndef BLOSC1_GEN_FIXTURE_SCRIPT
#error "BLOSC1_GEN_FIXTURE_SCRIPT must be defined by the build system"
#endif

struct fixture
{
  const char* name;
  uint8_t codec_id;
};

static const struct fixture k_fixtures[] = {
  { "lz4_noshuffle_ts4", CODEC_BLOSC_LZ4 },
  { "lz4_shuffle_ts4", CODEC_BLOSC_LZ4 },
  { "zstd_noshuffle_ts4", CODEC_BLOSC_ZSTD },
  { "lz4_noshuffle_ts1", CODEC_BLOSC_LZ4 },
  { "lz4_noshuffle_ts4_mb", CODEC_BLOSC_LZ4 },
  { "lz4_bitshuffle_ts4", CODEC_BLOSC_LZ4 },
};

static int
file_exists(const char* path)
{
  struct stat st;
  return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static int
ensure_fixtures(void)
{
  char path[512];
  int all_present = 1;
  for (size_t i = 0; i < sizeof k_fixtures / sizeof *k_fixtures; ++i) {
    snprintf(path,
             sizeof path,
             "%s/%s.blosc1",
             BLOSC1_FIXTURES_DIR,
             k_fixtures[i].name);
    if (!file_exists(path)) {
      all_present = 0;
      break;
    }
  }
  if (all_present)
    return 0;
  char cmd[1024];
  snprintf(cmd,
           sizeof cmd,
           "cd %s && uv run --script %s",
           BLOSC1_FIXTURES_DIR,
           BLOSC1_GEN_FIXTURE_SCRIPT);
  return system(cmd);
}

static int
slurp_file(const char* path, uint8_t** out_bytes, size_t* out_n)
{
  FILE* f = fopen(path, "rb");
  if (!f)
    return 1;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (sz < 0) {
    fclose(f);
    return 1;
  }
  uint8_t* buf = (uint8_t*)malloc((size_t)sz);
  if (!buf) {
    fclose(f);
    return 1;
  }
  if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
    free(buf);
    fclose(f);
    return 1;
  }
  fclose(f);
  *out_bytes = buf;
  *out_n = (size_t)sz;
  return 0;
}

// One fixture: host-parse the chunk, H2D the compressed bytes + the host
// fanout / op SOA arrays, run nvcomp + (bit)unshuffle, memcmp result.
static int
run_one(const struct fixture* fx, struct threadpool* pool, CUstream stream)
{
  char comp_path[512], raw_path[512];
  snprintf(
    comp_path, sizeof comp_path, "%s/%s.blosc1", BLOSC1_FIXTURES_DIR, fx->name);
  snprintf(
    raw_path, sizeof raw_path, "%s/%s.raw", BLOSC1_FIXTURES_DIR, fx->name);

  uint8_t* h_comp = NULL;
  uint8_t* h_raw = NULL;
  size_t comp_n = 0, raw_n = 0;
  EXPECT(slurp_file(comp_path, &h_comp, &comp_n) == 0);
  EXPECT(slurp_file(raw_path, &h_raw, &raw_n) == 0);

  CUdeviceptr d_comp = 0, d_decomp = 0;
  EXPECT(cuMemAlloc(&d_comp, comp_n) == CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_decomp, raw_n) == CUDA_SUCCESS);
  EXPECT(cuMemcpyHtoDAsync(d_comp, h_comp, comp_n, stream) == CUDA_SUCCESS);
  EXPECT(cuMemsetD8Async(d_decomp, 0, raw_n, stream) == CUDA_SUCCESS);

  // Host parse: builds host-resident fanout / op arrays. Pointer slots
  // hold device addresses (d_comp / d_decomp + per-substream offset).
  struct blosc1_host_chunk h_chunk = {
    .h_compressed = h_comp,
    .d_compressed = (void*)(uintptr_t)d_comp,
    .d_decompressed = (void*)(uintptr_t)d_decomp,
    .compressed_nbytes = (uint32_t)comp_n,
    .decompressed_nbytes = (uint32_t)raw_n,
    .codec_id = fx->codec_id,
  };

  struct blosc1_chunk_hdr h_hdr = { 0 };
  struct blosc1_chunk_counts h_counts = { 0 };
  struct blosc1_chunk_offsets h_offsets = { 0 };
  struct blosc1_host_scratch scratch = {
    .hdrs = &h_hdr,
    .counts = &h_counts,
    .offsets = &h_offsets,
  };

  // Worst-case substream count for one chunk: nblocks ≤ 16,
  // nstreams_per_block ≤ 8.
  const size_t kMaxSubs = 128;
  const void** zstd_comp_ptrs =
    (const void**)calloc(kMaxSubs, sizeof(*zstd_comp_ptrs));
  size_t* zstd_comp_sizes = (size_t*)calloc(kMaxSubs, sizeof(*zstd_comp_sizes));
  void** zstd_decomp_ptrs = (void**)calloc(kMaxSubs, sizeof(*zstd_decomp_ptrs));
  size_t* zstd_decomp_sizes =
    (size_t*)calloc(kMaxSubs, sizeof(*zstd_decomp_sizes));
  const void** lz4_comp_ptrs =
    (const void**)calloc(kMaxSubs, sizeof(*lz4_comp_ptrs));
  size_t* lz4_comp_sizes = (size_t*)calloc(kMaxSubs, sizeof(*lz4_comp_sizes));
  void** lz4_decomp_ptrs = (void**)calloc(kMaxSubs, sizeof(*lz4_decomp_ptrs));
  size_t* lz4_decomp_sizes =
    (size_t*)calloc(kMaxSubs, sizeof(*lz4_decomp_sizes));
  struct gpu_memcpy_op* memcpy_ops =
    (struct gpu_memcpy_op*)calloc(kMaxSubs, sizeof(*memcpy_ops));
  struct gpu_shuffle_op unsh = { 0 };
  struct gpu_shuffle_op bitunsh = { 0 };
  struct blosc1_totals h_totals = { 0 };

  struct blosc1_host_fanout host_zstd = {
    .comp_ptrs = zstd_comp_ptrs,
    .comp_sizes = zstd_comp_sizes,
    .decomp_ptrs = zstd_decomp_ptrs,
    .decomp_buf_sizes = zstd_decomp_sizes,
  };
  struct blosc1_host_fanout host_lz4 = {
    .comp_ptrs = lz4_comp_ptrs,
    .comp_sizes = lz4_comp_sizes,
    .decomp_ptrs = lz4_decomp_ptrs,
    .decomp_buf_sizes = lz4_decomp_sizes,
  };

  EXPECT(blosc1_host_parse(pool,
                           &h_chunk,
                           1,
                           scratch,
                           host_zstd,
                           host_lz4,
                           memcpy_ops,
                           &unsh,
                           &bitunsh,
                           &h_totals) == 0);
  EXPECT(h_hdr.err == 0);

  // Cross-check that the parser pulled the filter flags. Strings are
  // load-bearing here: a flag-extraction bug would otherwise show up
  // only as a wrong final memcmp, which is harder to localise.
  if (strstr(fx->name, "_shuffle_")) {
    EXPECT(h_hdr.shuffle == 1);
    EXPECT(h_hdr.bitshuffle == 0);
  } else if (strstr(fx->name, "_bitshuffle_")) {
    EXPECT(h_hdr.shuffle == 0);
    EXPECT(h_hdr.bitshuffle == 1);
  } else {
    EXPECT(h_hdr.shuffle == 0);
    EXPECT(h_hdr.bitshuffle == 0);
  }
  // Pure-codec fixtures (no per-block raw fallback expected at clevel=5
  // on this low-entropy data).
  EXPECT(h_totals.n_memcpy == 0);

  // H2D the host-built fanout / op arrays so nvcomp + the (bit)unshuffle
  // / memcpy launches can read them off-device. Each codec's contiguous
  // slice is exactly its `n_*` substreams; the rest of the kMaxSubs
  // capacity is unused.
  CUdeviceptr d_zstd_cp = 0, d_zstd_cs = 0, d_zstd_dp = 0, d_zstd_ds = 0;
  CUdeviceptr d_lz4_cp = 0, d_lz4_cs = 0, d_lz4_dp = 0, d_lz4_ds = 0;
  CUdeviceptr d_memcpy_ops = 0, d_unsh = 0, d_bitunsh = 0;
  EXPECT(cuMemAlloc(&d_zstd_cp, kMaxSubs * sizeof(void*)) == CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_zstd_cs, kMaxSubs * sizeof(size_t)) == CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_zstd_dp, kMaxSubs * sizeof(void*)) == CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_zstd_ds, kMaxSubs * sizeof(size_t)) == CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_lz4_cp, kMaxSubs * sizeof(void*)) == CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_lz4_cs, kMaxSubs * sizeof(size_t)) == CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_lz4_dp, kMaxSubs * sizeof(void*)) == CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_lz4_ds, kMaxSubs * sizeof(size_t)) == CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_memcpy_ops, kMaxSubs * sizeof(struct gpu_memcpy_op)) ==
         CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_unsh, sizeof(struct gpu_shuffle_op)) == CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_bitunsh, sizeof(struct gpu_shuffle_op)) == CUDA_SUCCESS);
  if (h_totals.n_zstd > 0) {
    EXPECT(cuMemcpyHtoDAsync(d_zstd_cp,
                             zstd_comp_ptrs,
                             h_totals.n_zstd * sizeof(void*),
                             stream) == CUDA_SUCCESS);
    EXPECT(cuMemcpyHtoDAsync(d_zstd_cs,
                             zstd_comp_sizes,
                             h_totals.n_zstd * sizeof(size_t),
                             stream) == CUDA_SUCCESS);
    EXPECT(cuMemcpyHtoDAsync(d_zstd_dp,
                             zstd_decomp_ptrs,
                             h_totals.n_zstd * sizeof(void*),
                             stream) == CUDA_SUCCESS);
    EXPECT(cuMemcpyHtoDAsync(d_zstd_ds,
                             zstd_decomp_sizes,
                             h_totals.n_zstd * sizeof(size_t),
                             stream) == CUDA_SUCCESS);
  }
  if (h_totals.n_lz4 > 0) {
    EXPECT(cuMemcpyHtoDAsync(
             d_lz4_cp, lz4_comp_ptrs, h_totals.n_lz4 * sizeof(void*), stream) ==
           CUDA_SUCCESS);
    EXPECT(cuMemcpyHtoDAsync(d_lz4_cs,
                             lz4_comp_sizes,
                             h_totals.n_lz4 * sizeof(size_t),
                             stream) == CUDA_SUCCESS);
    EXPECT(cuMemcpyHtoDAsync(d_lz4_dp,
                             lz4_decomp_ptrs,
                             h_totals.n_lz4 * sizeof(void*),
                             stream) == CUDA_SUCCESS);
    EXPECT(cuMemcpyHtoDAsync(d_lz4_ds,
                             lz4_decomp_sizes,
                             h_totals.n_lz4 * sizeof(size_t),
                             stream) == CUDA_SUCCESS);
  }
  if (h_hdr.shuffle) {
    EXPECT(cuMemcpyHtoDAsync(d_unsh, &unsh, sizeof unsh, stream) ==
           CUDA_SUCCESS);
  }
  if (h_hdr.bitshuffle) {
    EXPECT(cuMemcpyHtoDAsync(d_bitunsh, &bitunsh, sizeof bitunsh, stream) ==
           CUDA_SUCCESS);
  }
  EXPECT(cuStreamSynchronize(stream) == CUDA_SUCCESS);

  size_t max_substream_uncompressed = h_hdr.blocksize ? h_hdr.blocksize : raw_n;

  if (h_totals.n_zstd > 0) {
    struct decoder_zstd* z =
      decoder_zstd_create(h_totals.n_zstd, max_substream_uncompressed, raw_n);
    EXPECT(z);
    EXPECT(decoder_zstd_batch_device(z,
                                     stream,
                                     (const void**)(uintptr_t)d_zstd_cp,
                                     (size_t*)(uintptr_t)d_zstd_cs,
                                     (void**)(uintptr_t)d_zstd_dp,
                                     (size_t*)(uintptr_t)d_zstd_ds,
                                     h_totals.n_zstd) == 0);
    EXPECT(cuStreamSynchronize(stream) == CUDA_SUCCESS);
    decoder_zstd_destroy(z);
  }
  if (h_totals.n_lz4 > 0) {
    struct decoder_lz4* l =
      decoder_lz4_create(h_totals.n_lz4, max_substream_uncompressed, raw_n);
    EXPECT(l);
    EXPECT(decoder_lz4_batch_device(l,
                                    stream,
                                    (const void**)(uintptr_t)d_lz4_cp,
                                    (size_t*)(uintptr_t)d_lz4_cs,
                                    (void**)(uintptr_t)d_lz4_dp,
                                    (size_t*)(uintptr_t)d_lz4_ds,
                                    h_totals.n_lz4) == 0);
    EXPECT(cuStreamSynchronize(stream) == CUDA_SUCCESS);
    decoder_lz4_destroy(l);
  }

  CUdeviceptr d_scratch = 0;
  if (h_hdr.shuffle || h_hdr.bitshuffle)
    EXPECT(cuMemAlloc(&d_scratch, raw_n) == CUDA_SUCCESS);
  if (h_hdr.shuffle) {
    EXPECT(gpu_unshuffle_launch(stream,
                                (const struct gpu_shuffle_op*)(uintptr_t)d_unsh,
                                1,
                                (const void*)(uintptr_t)d_decomp,
                                (void*)(uintptr_t)d_scratch) == 0);
    EXPECT(cuStreamSynchronize(stream) == CUDA_SUCCESS);
  }
  if (h_hdr.bitshuffle) {
    EXPECT(gpu_bitunshuffle_launch(
             stream,
             (const struct gpu_shuffle_op*)(uintptr_t)d_bitunsh,
             1,
             (const void*)(uintptr_t)d_decomp,
             (void*)(uintptr_t)d_scratch) == 0);
    EXPECT(cuStreamSynchronize(stream) == CUDA_SUCCESS);
  }

  uint8_t* h_got = (uint8_t*)malloc(raw_n);
  EXPECT(h_got);
  EXPECT(cuMemcpyDtoH(h_got, d_decomp, raw_n) == CUDA_SUCCESS);
  EXPECT(memcmp(h_got, h_raw, raw_n) == 0);
  free(h_got);

  free(h_comp);
  free(h_raw);
  free(zstd_comp_ptrs);
  free(zstd_comp_sizes);
  free(zstd_decomp_ptrs);
  free(zstd_decomp_sizes);
  free(lz4_comp_ptrs);
  free(lz4_comp_sizes);
  free(lz4_decomp_ptrs);
  free(lz4_decomp_sizes);
  free(memcpy_ops);
  cuMemFree(d_comp);
  cuMemFree(d_decomp);
  cuMemFree(d_zstd_cp);
  cuMemFree(d_zstd_cs);
  cuMemFree(d_zstd_dp);
  cuMemFree(d_zstd_ds);
  cuMemFree(d_lz4_cp);
  cuMemFree(d_lz4_cs);
  cuMemFree(d_lz4_dp);
  cuMemFree(d_lz4_ds);
  cuMemFree(d_memcpy_ops);
  cuMemFree(d_unsh);
  cuMemFree(d_bitunsh);
  cuMemFree(d_scratch); // cuMemFree(0) is a no-op
  return 0;
}

// Run the full fixture set against three pool sizes (0 = serial, 1 =
// single worker, 4 = multi-worker). Catches any data race between
// phase-A counts and phase-C emits.
static int
test_all_fixtures(void)
{
  CUstream stream = 0;
  EXPECT(cuStreamCreate(&stream, CU_STREAM_DEFAULT) == CUDA_SUCCESS);
  const int pool_sizes[] = { 0, 1, 4 };
  for (size_t s = 0; s < sizeof pool_sizes / sizeof *pool_sizes; ++s) {
    struct threadpool* pool = threadpool_new(pool_sizes[s]);
    EXPECT(pool);
    log_info("pool nthreads=%d", pool_sizes[s]);
    for (size_t i = 0; i < sizeof k_fixtures / sizeof *k_fixtures; ++i) {
      log_info("  fixture: %s", k_fixtures[i].name);
      EXPECT(run_one(&k_fixtures[i], pool, stream) == 0);
    }
    threadpool_free(pool);
  }
  cuStreamDestroy(stream);
  return 0;
}

int
main(void)
{
  EXPECT(cuInit(0) == CUDA_SUCCESS);
  CUdevice dev = 0;
  EXPECT(cuDeviceGet(&dev, 0) == CUDA_SUCCESS);
  CUcontext ctx = NULL;
  EXPECT(cuDevicePrimaryCtxRetain(&ctx, dev) == CUDA_SUCCESS);
  EXPECT(cuCtxSetCurrent(ctx) == CUDA_SUCCESS);

  EXPECT(ensure_fixtures() == 0);
  RUN(test_all_fixtures);

  cuDevicePrimaryCtxRelease(dev);
  log_info("all tests passed");
  return 0;
}
