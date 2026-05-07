#include "decoder/bitshuffle.h"
#include "decoder/blosc1.h"
#include "decoder/decoder_lz4.h"
#include "decoder/decoder_memcpy.h"
#include "decoder/decoder_zstd.h"
#include "decoder/shuffle.h"
#include "expect.h"
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

static int
run_one(const struct fixture* fx, CUstream stream)
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

  struct blosc1_chunk_input h_in = { 0 };
  h_in.d_compressed = (const void*)(uintptr_t)d_comp;
  h_in.d_decompressed = (void*)(uintptr_t)d_decomp;
  h_in.compressed_nbytes = (uint32_t)comp_n;
  h_in.decompressed_nbytes = (uint32_t)raw_n;
  h_in.codec_id = fx->codec_id;

  CUdeviceptr d_in = 0, d_hdrs = 0, d_counts = 0, d_offsets = 0, d_totals = 0;
  EXPECT(cuMemAlloc(&d_in, sizeof h_in) == CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_hdrs, sizeof(struct blosc1_chunk_hdr)) == CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_counts, sizeof(struct blosc1_chunk_counts)) ==
         CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_offsets, sizeof(struct blosc1_chunk_offsets)) ==
         CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_totals, sizeof(struct blosc1_totals)) == CUDA_SUCCESS);
  EXPECT(cuMemcpyHtoDAsync(d_in, &h_in, sizeof h_in, stream) == CUDA_SUCCESS);

  EXPECT(blosc1_parse_and_count_launch(
           stream,
           (const struct blosc1_chunk_input*)(uintptr_t)d_in,
           (struct blosc1_chunk_hdr*)(uintptr_t)d_hdrs,
           (struct blosc1_chunk_counts*)(uintptr_t)d_counts,
           1) == 0);
  EXPECT(blosc1_scan_offsets_launch(
           stream,
           (const struct blosc1_chunk_counts*)(uintptr_t)d_counts,
           (struct blosc1_chunk_offsets*)(uintptr_t)d_offsets,
           (struct blosc1_totals*)(uintptr_t)d_totals,
           1) == 0);

  // Worst-case fanout: nblocks ≤ 16, nstreams_per_block ≤ 8.
  const size_t kMaxSubs = 128;
  CUdeviceptr d_zstd_subs = 0, d_lz4_subs = 0, d_memcpy_ops = 0;
  CUdeviceptr d_unsh = 0, d_bitunsh = 0;
  EXPECT(cuMemAlloc(&d_zstd_subs, kMaxSubs * sizeof(struct gpu_substream)) ==
         CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_lz4_subs, kMaxSubs * sizeof(struct gpu_substream)) ==
         CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_memcpy_ops, sizeof(struct gpu_memcpy_op)) ==
         CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_unsh, sizeof(struct gpu_shuffle_op)) == CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_bitunsh, sizeof(struct gpu_shuffle_op)) == CUDA_SUCCESS);

  EXPECT(blosc1_emit_fanout_launch(
           stream,
           (const struct blosc1_chunk_input*)(uintptr_t)d_in,
           (const struct blosc1_chunk_hdr*)(uintptr_t)d_hdrs,
           (const struct blosc1_chunk_offsets*)(uintptr_t)d_offsets,
           (struct gpu_substream*)(uintptr_t)d_zstd_subs,
           (struct gpu_substream*)(uintptr_t)d_lz4_subs,
           (struct gpu_memcpy_op*)(uintptr_t)d_memcpy_ops,
           (struct gpu_shuffle_op*)(uintptr_t)d_unsh,
           (struct gpu_shuffle_op*)(uintptr_t)d_bitunsh,
           1) == 0);

  struct blosc1_totals h_totals = { 0 };
  EXPECT(cuMemcpyDtoHAsync(&h_totals, d_totals, sizeof h_totals, stream) ==
         CUDA_SUCCESS);
  struct blosc1_chunk_hdr h_hdr = { 0 };
  EXPECT(cuMemcpyDtoHAsync(&h_hdr, d_hdrs, sizeof h_hdr, stream) ==
         CUDA_SUCCESS);
  EXPECT(cuStreamSynchronize(stream) == CUDA_SUCCESS);
  EXPECT(h_hdr.err == 0);

  // Pull the substream table back to host so we can hand it to the
  // existing host-array decoder API. Phase 5 will introduce a
  // device-array variant so this round-trip goes away.
  struct gpu_substream h_subs[128] = { 0 };
  size_t n_subs = 0;
  uint8_t which = 0; // 1 = zstd, 2 = lz4
  if (h_totals.n_zstd > 0) {
    EXPECT(h_totals.n_zstd <= kMaxSubs);
    EXPECT(cuMemcpyDtoH(h_subs,
                        d_zstd_subs,
                        h_totals.n_zstd * sizeof *h_subs) == CUDA_SUCCESS);
    n_subs = h_totals.n_zstd;
    which = 1;
  } else if (h_totals.n_lz4 > 0) {
    EXPECT(h_totals.n_lz4 <= kMaxSubs);
    EXPECT(cuMemcpyDtoH(h_subs, d_lz4_subs, h_totals.n_lz4 * sizeof *h_subs) ==
           CUDA_SUCCESS);
    n_subs = h_totals.n_lz4;
    which = 2;
  }
  EXPECT(n_subs > 0);

  const void** comp_ptrs = (const void**)malloc(n_subs * sizeof(void*));
  size_t* comp_sizes = (size_t*)malloc(n_subs * sizeof(size_t));
  void** decomp_ptrs = (void**)malloc(n_subs * sizeof(void*));
  size_t* decomp_sizes = (size_t*)malloc(n_subs * sizeof(size_t));
  EXPECT(comp_ptrs && comp_sizes && decomp_ptrs && decomp_sizes);
  for (size_t i = 0; i < n_subs; ++i) {
    comp_ptrs[i] = h_subs[i].d_src;
    comp_sizes[i] = h_subs[i].src_nbytes;
    decomp_ptrs[i] = h_subs[i].d_dst;
    decomp_sizes[i] = h_subs[i].dst_nbytes;
  }

  size_t max_substream_uncompressed = h_hdr.blocksize ? h_hdr.blocksize : raw_n;

  if (which == 1) {
    struct decoder_zstd* z =
      decoder_zstd_create(n_subs, max_substream_uncompressed);
    EXPECT(z);
    EXPECT(
      decoder_zstd_batch(
        z, stream, comp_ptrs, comp_sizes, decomp_ptrs, decomp_sizes, n_subs) ==
      0);
    EXPECT(cuStreamSynchronize(stream) == CUDA_SUCCESS);
    decoder_zstd_destroy(z);
  } else {
    struct decoder_lz4* l =
      decoder_lz4_create(n_subs, max_substream_uncompressed);
    EXPECT(l);
    EXPECT(
      decoder_lz4_batch(
        l, stream, comp_ptrs, comp_sizes, decomp_ptrs, decomp_sizes, n_subs) ==
      0);
    EXPECT(cuStreamSynchronize(stream) == CUDA_SUCCESS);
    decoder_lz4_destroy(l);
  }

  if (h_hdr.shuffle) {
    EXPECT(gpu_unshuffle_launch(stream,
                                (const struct gpu_shuffle_op*)(uintptr_t)d_unsh,
                                1,
                                h_hdr.blocksize) == 0);
    EXPECT(cuStreamSynchronize(stream) == CUDA_SUCCESS);
  }
  if (h_hdr.bitshuffle) {
    EXPECT(gpu_bitunshuffle_launch(
             stream,
             (const struct gpu_shuffle_op*)(uintptr_t)d_bitunsh,
             1,
             h_hdr.blocksize) == 0);
    EXPECT(cuStreamSynchronize(stream) == CUDA_SUCCESS);
  }

  uint8_t* h_got = (uint8_t*)malloc(raw_n);
  EXPECT(h_got);
  EXPECT(cuMemcpyDtoH(h_got, d_decomp, raw_n) == CUDA_SUCCESS);
  EXPECT(memcmp(h_got, h_raw, raw_n) == 0);
  free(h_got);

  free(comp_ptrs);
  free(comp_sizes);
  free(decomp_ptrs);
  free(decomp_sizes);
  free(h_comp);
  free(h_raw);
  cuMemFree(d_comp);
  cuMemFree(d_decomp);
  cuMemFree(d_in);
  cuMemFree(d_hdrs);
  cuMemFree(d_counts);
  cuMemFree(d_offsets);
  cuMemFree(d_totals);
  cuMemFree(d_zstd_subs);
  cuMemFree(d_lz4_subs);
  cuMemFree(d_memcpy_ops);
  cuMemFree(d_unsh);
  cuMemFree(d_bitunsh);
  return 0;
}

static int
test_all_fixtures(void)
{
  CUstream stream = 0;
  EXPECT(cuStreamCreate(&stream, CU_STREAM_DEFAULT) == CUDA_SUCCESS);
  for (size_t i = 0; i < sizeof k_fixtures / sizeof *k_fixtures; ++i) {
    log_info("fixture: %s", k_fixtures[i].name);
    EXPECT(run_one(&k_fixtures[i], stream) == 0);
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
