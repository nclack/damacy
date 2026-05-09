// Host-side blosc1 chunk-header parse. Replaces the device-side
// parse_and_count / scan_offsets / emit_fanout kernels.
#pragma once

#include "decoder/blosc1.h"
#include "decoder/decoder_memcpy.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct threadpool;

  struct blosc1_host_chunk
  {
    const uint8_t* h_compressed;
    void* d_compressed;
    void* d_decompressed;
    uint32_t compressed_nbytes;
    uint32_t decompressed_nbytes;
    uint8_t codec_id;
  };

  // Host-resident SOA fanout. Pointer slots hold device addresses.
  struct blosc1_host_fanout
  {
    const void** comp_ptrs;
    size_t* comp_sizes;
    void** decomp_ptrs;
    size_t* decomp_buf_sizes;
  };

  // Per-wave scratch, allocated once and reused. The first three arrays
  // are sized for DAMACY_MAX_CHUNKS_PER_WAVE; bstarts / block_ends are
  // sized cap * DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK and indexed
  // [chunk_idx * DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK + block_idx].
  // bstarts / block_ends are filled by the count phase and consumed by
  // the emit phase to avoid recomputing the per-block layout.
  struct blosc1_host_scratch
  {
    struct blosc1_chunk_hdr* hdrs;
    struct blosc1_chunk_counts* counts;
    struct blosc1_chunk_offsets* offsets;
    uint32_t* bstarts;
    uint32_t* block_ends;
  };

  struct blosc1_host_parse_args
  {
    struct threadpool* pool; // NULL → serial on caller
    const struct blosc1_host_chunk* chunks;
    uint32_t n_chunks;
    struct blosc1_host_scratch scratch;
    struct blosc1_host_fanout zstd;
    struct blosc1_host_fanout lz4;
    struct gpu_memcpy_op* memcpy_ops;
    struct gpu_shuffle_op* unshuffle_ops;
    struct gpu_shuffle_op* bitunshuffle_ops;
    struct blosc1_totals* out_totals;
  };

  // Returns 0 on success, non-zero on parse failure (also reflected in
  // out_totals->n_parse_errors).
  int blosc1_host_parse(const struct blosc1_host_parse_args* args);

  // Short string for blosc1_chunk_hdr.err (1..10). Returns "ok" for 0
  // and "unknown" for any out-of-range value. Lifetime is static.
  const char* blosc1_host_parse_err_str(uint8_t err);

#ifdef __cplusplus
}
#endif
