// blosc1 type definitions shared by the host parse (decoder/blosc1_host.{h,c})
// and the device-side fanout consumers (nvcomp + memcpy + (bit)unshuffle).
// The host parse fills the per-chunk hdr / counts / offsets and writes
// device pointers into the SOA fanout slots; the count totals drive the
// nvcomp batch sizes.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // SOA decompress fanout in nvcomp's expected layout. All four arrays
  // are device-resident, sized for the worst-case substream count of
  // the codec (max_chunks_per_wave * blosc-blocks/chunk * substreams/block).
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

  // Parsed-out blosc1 chunk header for CODEC_BLOSC_*; left mostly-zero
  // (with codec_id stamped) for non-blosc codecs. err is non-zero on
  // parse failure for the chunk (1..8 — see decoder/blosc1_host.c).
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

  // Per-chunk op counts. n_unshuffle / n_bitunshuffle are 0 or 1; the
  // others can grow to substream counts. The serial scan in the host
  // parse sums these into blosc1_totals (matching n_* names).
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

  // Wave-level totals + error counters. n_parse_errors is set by the
  // host parse (counted as chunks with a non-zero h.err); n_codec_errors
  // is set on-device by decoder_status_reduce_launch after each nvcomp
  // batch and read back via a 4-byte D2H at the end of post_decode.
  struct blosc1_totals
  {
    uint32_t n_zstd;
    uint32_t n_lz4;
    uint32_t n_memcpy;
    uint32_t n_unshuffle;
    uint32_t n_bitunshuffle;
    uint32_t n_parse_errors;
    uint32_t n_codec_errors;
  };

#ifdef __cplusplus
}
#endif
