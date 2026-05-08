#include "damacy_limits.h"
#include "decoder/blosc1.h"
#include "decoder/blosc1_host.h"
#include "decoder/decoder_memcpy.h"
#include "zarr/zarr_metadata.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define countof(arr) (sizeof(arr) / sizeof((arr)[0]))

// Worst-case substream count for one chunk: nblocks ≤ 32 *
// nstreams_per_block ≤ DAMACY_BLOSC_MAX_TYPESIZE.
#define K_MAX_SUBS \
  (DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK * DAMACY_BLOSC_MAX_TYPESIZE)

static void
try_one(const uint8_t* data, size_t size, uint8_t codec_id, uint32_t dec_nbytes)
{
  struct blosc1_host_chunk chunk = {
    .h_compressed = data,
    .d_compressed = NULL,
    .d_decompressed = NULL,
    .compressed_nbytes = (uint32_t)size,
    .decompressed_nbytes = dec_nbytes,
    .codec_id = codec_id,
  };
  struct blosc1_chunk_hdr hdr = { 0 };
  struct blosc1_chunk_counts counts = { 0 };
  struct blosc1_chunk_offsets offsets = { 0 };
  uint32_t bstarts[DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK] = { 0 };
  uint32_t block_ends[DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK] = { 0 };
  struct blosc1_host_scratch scratch = {
    .hdrs = &hdr,
    .counts = &counts,
    .offsets = &offsets,
    .bstarts = bstarts,
    .block_ends = block_ends,
  };
  // Fanout / op buffers carry pointer-typed slots that emit_one reads
  // back; allocate the worst-case footprint up front and leave them
  // zeroed so AddressSanitizer notices any out-of-bounds write.
  const void* zcp[K_MAX_SUBS] = { 0 };
  size_t zcs[K_MAX_SUBS] = { 0 };
  void* zdp[K_MAX_SUBS] = { 0 };
  size_t zds[K_MAX_SUBS] = { 0 };
  const void* lcp[K_MAX_SUBS] = { 0 };
  size_t lcs[K_MAX_SUBS] = { 0 };
  void* ldp[K_MAX_SUBS] = { 0 };
  size_t lds[K_MAX_SUBS] = { 0 };
  struct blosc1_host_fanout zfan = { zcp, zcs, zdp, zds };
  struct blosc1_host_fanout lfan = { lcp, lcs, ldp, lds };
  struct gpu_memcpy_op mops[K_MAX_SUBS] = { 0 };
  struct gpu_shuffle_op sop = { 0 };
  struct gpu_shuffle_op bop = { 0 };
  struct blosc1_totals totals = { 0 };

  (void)blosc1_host_parse(&(struct blosc1_host_parse_args){
    .pool = NULL,
    .chunks = &chunk,
    .n_chunks = 1,
    .scratch = scratch,
    .zstd = zfan,
    .lz4 = lfan,
    .memcpy_ops = mops,
    .unshuffle_ops = &sop,
    .bitunshuffle_ops = &bop,
    .out_totals = &totals,
  });
}

static uint32_t
read_u32_le(const uint8_t* p)
{
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

int
LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
  // Honest dec_nbytes derived from the header so the success path is
  // reachable when the input is a valid blosc1 chunk (header.cbytes ==
  // size and header.nbytes == derived dec_nbytes).
  if (size >= 16) {
    const uint32_t honest_dec = read_u32_le(data + 4);
    try_one(data, size, CODEC_BLOSC_LZ4, honest_dec);
    try_one(data, size, CODEC_BLOSC_ZSTD, honest_dec);
  }

  // Adversarial decompressed_nbytes drives the err=3 branch and gives
  // the size-mismatch reject paths something to chew on.
  static const uint32_t mismatched[] = {
    0u, 1u, 64u, 1024u, 65536u, 0xffffffffu,
  };
  for (size_t i = 0; i < countof(mismatched); ++i) {
    try_one(data, size, CODEC_BLOSC_LZ4, mismatched[i]);
    try_one(data, size, CODEC_BLOSC_ZSTD, mismatched[i]);
  }

  // The CODEC_NONE / CODEC_ZSTD / unsupported branches are size-only
  // (no header read), but include them so a regression that suddenly
  // dereferences h_compressed there gets caught.
  try_one(data, size, CODEC_NONE, (uint32_t)size);
  try_one(data, size, CODEC_ZSTD, (uint32_t)size);
  try_one(data, size, 99u, (uint32_t)size);

  // n_chunks == 0 short-circuits past every per-chunk allocation; keep
  // the branch warm.
  if (size == 0) {
    struct blosc1_totals totals = { 0 };
    (void)blosc1_host_parse(&(struct blosc1_host_parse_args){
      .pool = NULL,
      .chunks = NULL,
      .n_chunks = 0,
      .out_totals = &totals,
    });
  }

  return 0;
}
