// blosc1_host_parse → nvcomp → (bit)unshuffle → memcmp against .raw.

#include "damacy_limits.h"
#include "decoder/bitshuffle.h"
#include "decoder/blosc1.h"
#include "decoder/blosc1_host.h"
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
  { "zstd_noshuffle_ts4", CODEC_BLOSC_ZSTD },
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
  uint32_t bstarts_slot[DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK] = { 0 };
  uint32_t block_ends_slot[DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK] = { 0 };
  struct blosc1_host_scratch scratch = {
    .hdrs = &h_hdr,
    .counts = &h_counts,
    .offsets = &h_offsets,
    .bstarts = bstarts_slot,
    .block_ends = block_ends_slot,
  };

  // Worst-case substream count for one chunk: each block emits exactly
  // one substream, so nblocks ≤ DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK (32).
  // Leave 128 as a harmless over-allocation.
  const size_t kMaxSubs = 128;
  const void** zstd_comp_ptrs =
    (const void**)calloc(kMaxSubs, sizeof(*zstd_comp_ptrs));
  size_t* zstd_comp_sizes = (size_t*)calloc(kMaxSubs, sizeof(*zstd_comp_sizes));
  void** zstd_decomp_ptrs = (void**)calloc(kMaxSubs, sizeof(*zstd_decomp_ptrs));
  size_t* zstd_decomp_sizes =
    (size_t*)calloc(kMaxSubs, sizeof(*zstd_decomp_sizes));
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

  EXPECT(blosc1_host_parse(&(struct blosc1_host_parse_args){
           .pool = pool,
           .chunks = &h_chunk,
           .n_chunks = 1,
           .scratch = scratch,
           .zstd = host_zstd,
           .memcpy_ops = memcpy_ops,
           .unshuffle_ops = &unsh,
           .bitunshuffle_ops = &bitunsh,
           .out_totals = &h_totals,
         }) == 0);
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
  // Fixtures are clevel=5 on low-entropy data; no raw fallback expected.
  EXPECT(h_totals.n_memcpy == 0);

  CUdeviceptr d_zstd_cp = 0, d_zstd_cs = 0, d_zstd_dp = 0, d_zstd_ds = 0;
  CUdeviceptr d_memcpy_ops = 0, d_unsh = 0, d_bitunsh = 0;
  EXPECT(cuMemAlloc(&d_zstd_cp, kMaxSubs * sizeof(void*)) == CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_zstd_cs, kMaxSubs * sizeof(size_t)) == CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_zstd_dp, kMaxSubs * sizeof(void*)) == CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_zstd_ds, kMaxSubs * sizeof(size_t)) == CUDA_SUCCESS);
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
    decoder_zstd_destroy(z, 0);
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
  free(memcpy_ops);
  cuMemFree(d_comp);
  cuMemFree(d_decomp);
  cuMemFree(d_zstd_cp);
  cuMemFree(d_zstd_cs);
  cuMemFree(d_zstd_dp);
  cuMemFree(d_zstd_ds);
  cuMemFree(d_memcpy_ops);
  cuMemFree(d_unsh);
  cuMemFree(d_bitunsh);
  cuMemFree(d_scratch); // cuMemFree(0) is a no-op
  return 0;
}

// Build a minimal blosc1 chunk header in h[0..16) with compformat
// matching codec_id. Caller mutates fields after to trigger err codes.
static void
build_blosc1_header(uint8_t* h,
                    uint8_t codec_id,
                    uint32_t nbytes,
                    uint32_t blocksize,
                    uint32_t cbytes,
                    uint8_t typesize)
{
  h[0] = 0;
  h[1] = 0;
  // CODEC_BLOSC_ZSTD => compformat 4. Other codec values are rejected at
  // parse with err=8 before compformat is inspected.
  (void)codec_id;
  uint8_t compformat = 4u;
  h[2] = (uint8_t)((compformat & 0x07u) << 5);
  h[3] = typesize;
  for (int i = 0; i < 4; ++i)
    h[4 + i] = (uint8_t)((nbytes >> (8 * i)) & 0xffu);
  for (int i = 0; i < 4; ++i)
    h[8 + i] = (uint8_t)((blocksize >> (8 * i)) & 0xffu);
  for (int i = 0; i < 4; ++i)
    h[12 + i] = (uint8_t)((cbytes >> (8 * i)) & 0xffu);
}

// Run blosc1_host_parse over a single synthetic chunk and assert it
// returns the expected err code in totals.n_parse_errors and hdrs[0].err.
// Op buffers are unused on the failure path so single-slot stack arrays
// are sufficient.
static int
expect_parse_err(uint8_t codec_id,
                 const uint8_t* hdr,
                 uint32_t compressed_nbytes,
                 uint32_t decompressed_nbytes,
                 uint8_t expected_err,
                 struct threadpool* pool)
{
  struct blosc1_host_chunk chunk = {
    .h_compressed = hdr,
    .d_compressed = NULL,
    .d_decompressed = NULL,
    .compressed_nbytes = compressed_nbytes,
    .decompressed_nbytes = decompressed_nbytes,
    .codec_id = codec_id,
  };
  struct blosc1_chunk_hdr h = { 0 };
  struct blosc1_chunk_counts c = { 0 };
  struct blosc1_chunk_offsets o = { 0 };
  uint32_t bstarts_slot[DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK] = { 0 };
  uint32_t block_ends_slot[DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK] = { 0 };
  struct blosc1_host_scratch scratch = {
    .hdrs = &h,
    .counts = &c,
    .offsets = &o,
    .bstarts = bstarts_slot,
    .block_ends = block_ends_slot,
  };
  const void* zcp = NULL;
  size_t zcs = 0;
  void* zdp = NULL;
  size_t zds = 0;
  struct blosc1_host_fanout zfan = { &zcp, &zcs, &zdp, &zds };
  struct gpu_memcpy_op mop = { 0 };
  struct gpu_shuffle_op sop = { 0 };
  struct gpu_shuffle_op bop = { 0 };
  struct blosc1_totals tot = { 0 };
  int rc = blosc1_host_parse(&(struct blosc1_host_parse_args){
    .pool = pool,
    .chunks = &chunk,
    .n_chunks = 1,
    .scratch = scratch,
    .zstd = zfan,
    .memcpy_ops = &mop,
    .unshuffle_ops = &sop,
    .bitunshuffle_ops = &bop,
    .out_totals = &tot,
  });
  EXPECT(rc != 0);
  EXPECT(tot.n_parse_errors == 1);
  EXPECT(h.err == expected_err);
  return 0;
}

// Spot-check the err-code surface (1..10). Comprehensive coverage is
// expected to come from the fuzz harness tracked in the issue.
static int
test_bad_header(void)
{
  struct threadpool* pool = threadpool_new(0);
  EXPECT(pool);
  uint8_t hdr[16];

  // err=1: compressed_nbytes < 16 (header read short-circuits)
  build_blosc1_header(hdr, CODEC_BLOSC_ZSTD, 4096, 1024, 100, 4);
  EXPECT(expect_parse_err(CODEC_BLOSC_ZSTD, hdr, 8, 4096, 1, pool) == 0);

  // err=2: header.blocksize == 0
  build_blosc1_header(hdr, CODEC_BLOSC_ZSTD, 4096, 0, 100, 4);
  EXPECT(expect_parse_err(CODEC_BLOSC_ZSTD, hdr, 100, 4096, 2, pool) == 0);

  // err=3: header.nbytes != decompressed_nbytes
  build_blosc1_header(hdr, CODEC_BLOSC_ZSTD, 4096, 1024, 100, 4);
  EXPECT(expect_parse_err(CODEC_BLOSC_ZSTD, hdr, 100, 8192, 3, pool) == 0);

  // err=4: header.cbytes != compressed_nbytes
  build_blosc1_header(hdr, CODEC_BLOSC_ZSTD, 4096, 1024, 100, 4);
  EXPECT(expect_parse_err(CODEC_BLOSC_ZSTD, hdr, 200, 4096, 4, pool) == 0);

  // err=6: typesize == 0
  build_blosc1_header(hdr, CODEC_BLOSC_ZSTD, 4096, 1024, 100, 0);
  EXPECT(expect_parse_err(CODEC_BLOSC_ZSTD, hdr, 100, 4096, 6, pool) == 0);

  // err=7: header.compformat doesn't match codec_id. Build a zstd
  // header then overwrite compformat to the legacy lz4 value (1) so
  // it no longer matches BLOSC_ZSTD's expected 4.
  build_blosc1_header(hdr, CODEC_BLOSC_ZSTD, 4096, 1024, 100, 4);
  hdr[2] = (uint8_t)((1u & 0x07u) << 5);
  EXPECT(expect_parse_err(CODEC_BLOSC_ZSTD, hdr, 100, 4096, 7, pool) == 0);

  // err=8: unsupported codec_id
  build_blosc1_header(hdr, CODEC_BLOSC_ZSTD, 4096, 1024, 100, 4);
  EXPECT(expect_parse_err(99u, hdr, 100, 4096, 8, pool) == 0);

  // err=8: CODEC_BLOSC_LZ4 must also route to err=8. The planner
  // rejects LZ4 upstream, but pin the host loader's defense-in-depth
  // behavior in case that early rejection is ever removed or bypassed.
  build_blosc1_header(hdr, CODEC_BLOSC_ZSTD, 4096, 1024, 100, 4);
  EXPECT(expect_parse_err(CODEC_BLOSC_LZ4, hdr, 100, 4096, 8, pool) == 0);

  // err=9: bstart out of range. Build a valid 16+4*nblocks header
  // followed by bstart entries that point past cbytes. nblocks=4 →
  // cbytes=32 (header 16 + 4 bstarts * 4). bstarts[0] = 0xffffffff
  // is far past cbytes; the loader must reject before walk_count.
  uint8_t buf9[32];
  memset(buf9, 0, sizeof buf9);
  build_blosc1_header(buf9, CODEC_BLOSC_ZSTD, 4096, 1024, 32, 4);
  for (int i = 0; i < 4; ++i) {
    buf9[16 + i * 4 + 0] = 0xff;
    buf9[16 + i * 4 + 1] = 0xff;
    buf9[16 + i * 4 + 2] = 0xff;
    buf9[16 + i * 4 + 3] = 0xff;
  }
  EXPECT(expect_parse_err(CODEC_BLOSC_ZSTD, buf9, 32, 4096, 9, pool) == 0);

  // err=9 boundary: bs == cbytes (one past last byte; can't fit a
  // 4-byte prefix, must be rejected).
  uint8_t buf9b[32];
  memset(buf9b, 0, sizeof buf9b);
  build_blosc1_header(buf9b, CODEC_BLOSC_ZSTD, 4096, 1024, 32, 4);
  for (int i = 0; i < 4; ++i) {
    const uint32_t bs = 32u; // cbytes
    buf9b[16 + i * 4 + 0] = (uint8_t)(bs & 0xffu);
    buf9b[16 + i * 4 + 1] = (uint8_t)((bs >> 8) & 0xffu);
    buf9b[16 + i * 4 + 2] = (uint8_t)((bs >> 16) & 0xffu);
    buf9b[16 + i * 4 + 3] = (uint8_t)((bs >> 24) & 0xffu);
  }
  EXPECT(expect_parse_err(CODEC_BLOSC_ZSTD, buf9b, 32, 4096, 9, pool) == 0);

  // err=9 boundary: bs == payload_lo - 1 (one byte below the bstart
  // table's end; would land inside the table).
  uint8_t buf9c[32];
  memset(buf9c, 0, sizeof buf9c);
  build_blosc1_header(buf9c, CODEC_BLOSC_ZSTD, 4096, 1024, 32, 4);
  for (int i = 0; i < 4; ++i) {
    const uint32_t bs = 16u + 4u * 4u - 1u; // payload_lo - 1
    buf9c[16 + i * 4 + 0] = (uint8_t)(bs & 0xffu);
    buf9c[16 + i * 4 + 1] = (uint8_t)((bs >> 8) & 0xffu);
    buf9c[16 + i * 4 + 2] = (uint8_t)((bs >> 16) & 0xffu);
    buf9c[16 + i * 4 + 3] = (uint8_t)((bs >> 24) & 0xffu);
  }
  EXPECT(expect_parse_err(CODEC_BLOSC_ZSTD, buf9c, 32, 4096, 9, pool) == 0);

  // err=10: header.nbytes > DAMACY_BLOSC_MAX_CHUNK_UNCOMPRESSED_BYTES.
  // Use UINT32_MAX so the prior nblocks ceil-div would have wrapped
  // to 0 if the cap weren't enforced.
  build_blosc1_header(hdr, CODEC_BLOSC_ZSTD, 0xffffffffu, 2, 100, 4);
  EXPECT(expect_parse_err(CODEC_BLOSC_ZSTD, hdr, 100, 0xffffffffu, 10, pool) ==
         0);

  // err=11: truncated block cbytes prefix. Pass err=9 (bstarts in
  // [payload_lo, cbytes)) but place two bstarts only 1 byte apart so
  // the first block has end - bstart = 1 < 4 and walk_count's prefix
  // read would underrun. nbytes=2048, blocksize=1024 → nblocks=2,
  // payload_lo=24. bstarts={24,25}, cbytes=32. After sort,
  // block_ends[0]=25, so for block 0: end - 4 = 21 < bs=24 ⇒ err=11.
  uint8_t buf11[32];
  memset(buf11, 0, sizeof buf11);
  build_blosc1_header(buf11, CODEC_BLOSC_ZSTD, 2048, 1024, 32, 4);
  const uint32_t bs11[2] = { 24u, 25u };
  for (int i = 0; i < 2; ++i) {
    buf11[16 + i * 4 + 0] = (uint8_t)(bs11[i] & 0xffu);
    buf11[16 + i * 4 + 1] = (uint8_t)((bs11[i] >> 8) & 0xffu);
    buf11[16 + i * 4 + 2] = (uint8_t)((bs11[i] >> 16) & 0xffu);
    buf11[16 + i * 4 + 3] = (uint8_t)((bs11[i] >> 24) & 0xffu);
  }
  EXPECT(expect_parse_err(CODEC_BLOSC_ZSTD, buf11, 32, 2048, 11, pool) == 0);

  // Spot-check the stringifier surface too.
  EXPECT(strcmp(blosc1_host_parse_err_str(0), "ok") == 0);
  EXPECT(strcmp(blosc1_host_parse_err_str(10),
                "header.nbytes > DAMACY_BLOSC_MAX_CHUNK_UNCOMPRESSED_BYTES") ==
         0);
  EXPECT(strcmp(blosc1_host_parse_err_str(11),
                "truncated block cbytes prefix") == 0);
  EXPECT(strcmp(blosc1_host_parse_err_str(99), "unknown") == 0);

  threadpool_free(pool);
  return 0;
}

// Single deterministic path: pool=0 (serial). Multi-worker behavior is
// not exercised here — see threadpool's own tests for that — to avoid
// false confidence from a non-determ smoke test of the fork-join.
static int
test_all_fixtures(void)
{
  CUstream stream = 0;
  EXPECT(cuStreamCreate(&stream, CU_STREAM_DEFAULT) == CUDA_SUCCESS);
  struct threadpool* pool = threadpool_new(0);
  EXPECT(pool);
  for (size_t i = 0; i < sizeof k_fixtures / sizeof *k_fixtures; ++i) {
    log_info("fixture: %s", k_fixtures[i].name);
    EXPECT(run_one(&k_fixtures[i], pool, stream) == 0);
  }
  threadpool_free(pool);
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

  RUN(test_bad_header);
  EXPECT(ensure_fixtures() == 0);
  RUN(test_all_fixtures);

  cuDevicePrimaryCtxRelease(dev);
  log_info("all tests passed");
  return 0;
}
