#pragma once

#include "decoder/decoder_memcpy.h"

#include <cuda.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct gpu_substream
  {
    const void* d_src;
    void* d_dst;
    uint32_t src_nbytes;
    uint32_t dst_nbytes;
  };

  struct gpu_shuffle_op
  {
    void* d_buf;
    uint32_t blocksize;
    uint32_t typesize;
    uint32_t nblocks_full;
    uint32_t tail_nbytes;
  };

  // One per chunk in the wave. Driven by the planner output (codec_id,
  // sizes) and resolved device pointers into the wave's compressed and
  // decompressed arenas.
  struct blosc1_chunk_input
  {
    const void* d_compressed;
    void* d_decompressed;
    uint32_t compressed_nbytes;
    uint32_t decompressed_nbytes;
    uint8_t codec_id;
    uint8_t _pad[7];
  };

  // Parsed-out blosc1 chunk header. Filled by blosc1_parse_and_count for
  // CODEC_BLOSC_*; left mostly-zero (with codec_id stamped) for non-blosc
  // codecs. err is non-zero on parse failure for the chunk.
  struct blosc1_chunk_hdr
  {
    uint8_t codec_id;
    uint8_t typesize;
    uint8_t shuffle;
    uint8_t bitshuffle;
    uint8_t memcpyed;
    uint8_t compformat;
    uint8_t err;
    uint8_t _pad;
    uint32_t nbytes;
    uint32_t blocksize;
    uint32_t cbytes;
    uint32_t nblocks;
  };

  struct blosc1_chunk_counts
  {
    uint32_t n_zstd;
    uint32_t n_lz4;
    uint32_t n_memcpy;
    uint32_t has_unshuffle;
    uint32_t has_bitunshuffle;
  };

  struct blosc1_chunk_offsets
  {
    uint32_t zstd_off;
    uint32_t lz4_off;
    uint32_t memcpy_off;
    uint32_t unshuffle_off;
    uint32_t bitunshuffle_off;
  };

  struct blosc1_totals
  {
    uint32_t n_zstd;
    uint32_t n_lz4;
    uint32_t n_memcpy;
    uint32_t n_unshuffle;
    uint32_t n_bitunshuffle;
  };

  int blosc1_parse_and_count_launch(CUstream stream,
                                    const struct blosc1_chunk_input* d_inputs,
                                    struct blosc1_chunk_hdr* d_hdrs,
                                    struct blosc1_chunk_counts* d_counts,
                                    uint32_t n_chunks);

  int blosc1_scan_offsets_launch(CUstream stream,
                                 const struct blosc1_chunk_counts* d_counts,
                                 struct blosc1_chunk_offsets* d_offsets,
                                 struct blosc1_totals* d_totals,
                                 uint32_t n_chunks);

  int blosc1_emit_fanout_launch(CUstream stream,
                                const struct blosc1_chunk_input* d_inputs,
                                const struct blosc1_chunk_hdr* d_hdrs,
                                const struct blosc1_chunk_offsets* d_offsets,
                                struct gpu_substream* d_zstd_subs,
                                struct gpu_substream* d_lz4_subs,
                                struct gpu_memcpy_op* d_memcpy_ops,
                                struct gpu_shuffle_op* d_unshuffle_ops,
                                struct gpu_shuffle_op* d_bitunshuffle_ops,
                                uint32_t n_chunks);

#ifdef __cplusplus
}
#endif
