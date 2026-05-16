// GPU blosc1 chunk-header parse. Handles only what needs device-resident
// bytes: the per-chunk memcpyed flag (varies per chunk) and the per-block
// bstarts + 4B cb prefixes. FILL/NONE/ZSTD whole-chunk ops and the
// array-invariant assemble shuffle fields are pre-filled on the host
// from sample_plan.layout; the routing layer falls back to the host
// parser if any BLOSC_ZSTD chunk's sample lacks a probed layout.
//
// Two kernels run on the same stream:
//   A (chunk_memcpyed_scan): one thread per BLOSC_ZSTD chunk. Reads the
//     header flags, validates cbytes, emits a single memcpy op for
//     memcpyed chunks, sets the d_is_memcpyed bitset and the chunk's
//     assemble shuffle fields.
//   B (block_fanout): one thread per blosc block across non-memcpyed
//     BLOSC_ZSTD chunks. Walks bstart + cb prefix, classifies raw vs
//     zstd, warp-aggregates slot reservations into the SOAs.
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

    // Subset of d_chunks that are BLOSC_ZSTD (indices into d_chunks).
    const uint32_t* d_blosc_chunk_indices;
    uint32_t n_blosc_zstd_chunks;

    // Packed (chunk_idx << 16) | block_local_idx, length
    // n_blosc_zstd_blocks. Built host-side from the per-sample
    // layout.nblocks.
    const uint32_t* d_block_chunk_map;
    uint32_t n_blosc_zstd_blocks;

    // Bitset cleared by caller before launch; Kernel A sets a bit for
    // each chunk whose memcpyed flag is on. Kernel B reads it to skip
    // those chunks' blocks. Sized for the wave's total chunk count.
    uint32_t* d_is_memcpyed;

    // Output SOAs (device-resident, sized for the wave's worst case).
    // Counters must be pre-initialized by the caller to the count of
    // already host-emitted slots (FILL/NONE/ZSTD whole-chunk ops); the
    // kernels atomicAdd on top of that.
    struct nvcomp_fanout zstd;
    struct gpu_memcpy_op* d_memcpy_ops;
    struct assemble_chunk* d_assemble_chunks;
    uint32_t* d_n_zstd;
    uint32_t* d_n_memcpy;
    // First parse error (1..). Codes emitted by these kernels: 4
    // (cbytes mismatch), 9 (bstart range), 11 (cb overflow). Other
    // historical codes are pre-empted by the host routing.
    uint32_t* d_parse_err;
  };

  // Launches Kernel A then Kernel B on `stream`. No-op when
  // n_blosc_zstd_chunks == 0. Returns 0 on launch success.
  int blosc1_parse_launch(CUstream stream,
                          const struct blosc1_parse_args* args);

#ifdef __cplusplus
}
#endif
