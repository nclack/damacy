#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct nvcomp_fanout
  {
    const void** d_comp_ptrs;
    size_t* d_comp_sizes;
    void** d_decomp_ptrs;
    size_t* d_decomp_buf_sizes;
  };

  struct gpu_shuffle_op
  {
    void* d_buf;
    uint32_t blocksize;
    uint32_t typesize;
    uint32_t nblocks_full;
    uint32_t tail_nbytes;
  };

  struct blosc1_chunk_hdr
  {
    uint8_t codec_id;
    uint8_t typesize;
    uint8_t shuffle;
    uint8_t bitshuffle;
    uint8_t memcpyed;
    uint8_t compformat;
    uint8_t err; // 0 == ok; 1..8 see decoder/blosc1_host.c
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
    uint32_t n_unshuffle;
    uint32_t n_bitunshuffle;
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
    uint32_t n_parse_errors; // host parse
    uint32_t n_codec_errors; // device, via decoder_status_reduce
  };

#ifdef __cplusplus
}
#endif
