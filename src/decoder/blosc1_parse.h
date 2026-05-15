// GPU blosc1 chunk-header parse. Sibling of blosc1_host.* — replaces
// the host parse + the H2D of fanout/memcpy_ops on the GPU-parse path.
// Reads device-resident compressed bytes, walks each chunk's blosc1
// header + bstarts + per-block 4B prefixes, and populates the existing
// per-wave SOAs in place: d_zstd_fan, d_memcpy_ops, d_assemble_chunks.
//
// Zstd-only by design (substreams-per-block ≡ 1, no chain walk). The
// kernel handles CODEC_BLOSC_ZSTD, CODEC_ZSTD, CODEC_NONE, and
// CODEC_FILL; mixed-codec waves are routed onto the host parse instead.
#pragma once

#include "decoder/blosc1.h"         // struct nvcomp_fanout
#include "decoder/decoder_memcpy.h" // struct gpu_memcpy_op

#include <stdint.h>

typedef struct CUstream_st* CUstream;

#ifdef __cplusplus
extern "C"
{
#endif

  struct assemble_chunk;
  struct sample_plan;

  // Per-chunk kernel input. Built on host from chunk_plan + sample_plan,
  // uploaded to device alongside d_assemble_chunks. 16 bytes/chunk;
  // ≤8 KB for a 512-chunk wave.
  struct gpu_parse_chunk
  {
    uint32_t compressed_offset; // byte offset into d_compressed
    uint32_t compressed_nbytes;
    uint32_t decompressed_offset; // byte offset into d_decompressed
    uint32_t decompressed_nbytes;
    uint16_t sample_idx_in_batch; // index into d_sample_plans
    uint8_t codec_id;             // enum compression_codec
    uint8_t is_fill;              // 1 = absent chunk; kernel skips it
  };

  // Launch arguments. All pointers are device pointers unless noted.
  struct blosc1_parse_args
  {
    const uint8_t* d_compressed; // wave->dev_compressed
    uint8_t* d_decompressed;     // wave->dev_decompressed
    const struct gpu_parse_chunk* d_chunks;
    const struct sample_plan* d_sample_plans;
    uint32_t n_chunks;

    // Outputs (device-resident, sized for the wave's worst case).
    struct nvcomp_fanout zstd;
    struct gpu_memcpy_op* d_memcpy_ops;
    struct assemble_chunk* d_assemble_chunks;
    uint32_t* d_n_zstd;   // single-int counter, atomic-added
    uint32_t* d_n_memcpy; // single-int counter, atomic-added
    // Captures first parse error (1..) so the host can surface a clear
    // DAMACY_DECODE. Atomic CAS-to-1, but only the FIRST chunk to fail
    // writes anything meaningful; subsequent failures leave it as-is.
    uint32_t* d_parse_err;
  };

  // One CUDA block per chunk; up to DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK
  // threads per block. Returns 0 on launch success.
  int blosc1_parse_launch(CUstream stream,
                          const struct blosc1_parse_args* args);

#ifdef __cplusplus
}
#endif
