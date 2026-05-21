// Per-array blosc1 chunk layout, probed once from the first 16 bytes of
// any non-fill chunk. Lets the planner and wave pool size their
// decoder/fanout caps to exactly what the array needs (no first-wave
// grow event) and feeds blocksize / nblocks / shuffle into the GPU parse
// kernels via sample_plan.layout.
//
// Probing is best-effort: failures leave the caller on the existing
// observe-and-grow path with the initial-floor caps.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct store;

  struct chunk_layout
  {
    uint8_t codec_id;   // CODEC_BLOSC_ZSTD currently the only probed codec
    uint8_t typesize;   // bytes per element (1..8)
    uint8_t shuffle;    // byte shuffle flag
    uint8_t bitshuffle; // bit shuffle flag
    uint8_t dont_split; // header flag bit 4; always 1 for blosc1-zstd
    uint8_t memcpyed;   // header flag bit 1
    uint32_t blocksize; // bytes per blosc block
    uint32_t nbytes;    // uncompressed bytes per chunk
    uint32_t nblocks;   // == ceil(nbytes / blocksize); <= MAX_BLOCKS_PER_CHUNK
  };

  // Read the 16-byte blosc1 header at first_chunk_off in shard_path and
  // populate *out. Returns 0 on success, non-zero on IO failure, on a
  // malformed header, or for codecs that don't carry a blosc1 header
  // (CODEC_NONE / CODEC_ZSTD). On non-zero return *out is left unset.
  int zarr_chunk_layout_probe(struct store* s,
                              const char* shard_path,
                              uint64_t first_chunk_off,
                              uint32_t first_chunk_cbytes,
                              uint8_t codec_id,
                              uint32_t max_substreams_per_chunk,
                              struct chunk_layout* out);

#ifdef __cplusplus
}
#endif
