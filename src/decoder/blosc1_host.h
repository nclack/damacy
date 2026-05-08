// Host-side blosc1 chunk-header parse. Reads the compressed bytes from
// pinned host memory while the bulk H2D streams them to device, walks the
// blosc1 16-byte header + bstarts table + per-substream int32 prefixes,
// and emits per-codec nvcomp fanout SOAs + memcpy / shuffle op arrays
// directly into pinned-host buffers. The caller H2D's those buffers onto
// stream_h2d after parse returns; nvcomp / kernel launches then read them
// off-device.
//
// Replaces the device-side blosc1_parse_and_count / scan_offsets /
// emit_fanout kernels and the host stall they required.
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

  // One per chunk in the wave. The host pointer is read by the parse
  // (blosc1 header + bstarts + per-substream int32 prefixes); the device
  // pointers are written into the fanout SOAs as base + per-substream
  // offset (the host_slab and dev_compressed arenas are sized identically
  // and H2D'd byte-for-byte, so offsets are mechanically equivalent).
  struct blosc1_host_chunk
  {
    const uint8_t* h_compressed; // pinned host
    void* d_compressed;          // device base + same offset
    void* d_decompressed;        // device base + per-chunk arena offset
    uint32_t compressed_nbytes;
    uint32_t decompressed_nbytes;
    uint8_t codec_id;
    uint8_t _pad[7];
  };

  // SOA nvcomp fanout in HOST memory. Layout matches `nvcomp_fanout`; the
  // pointer slots hold device addresses (chunks[i].d_compressed +
  // sub_offset / chunks[i].d_decompressed + sub_offset).
  struct blosc1_host_fanout
  {
    const void** comp_ptrs;
    size_t* comp_sizes;
    void** decomp_ptrs;
    size_t* decomp_buf_sizes;
  };

  // Scratch arrays sized for one wave (DAMACY_MAX_CHUNKS_PER_WAVE entries
  // each). Allocated once at wave_init; reused per call without zeroing
  // (each parse re-fills the per-chunk records it touches).
  struct blosc1_host_scratch
  {
    struct blosc1_chunk_hdr* hdrs;
    struct blosc1_chunk_counts* counts;
    struct blosc1_chunk_offsets* offsets;
  };

  // Parse n_chunks blosc1 chunks on `pool`, fill the fanout / ops arrays,
  // produce wave-level totals.
  //
  // pool may be NULL (or have nworkers == 0) — both fall through to
  // serial execution on the calling thread.
  //
  // out_totals is fully written: count fields from the per-chunk scan
  // and n_parse_errors from the count of chunks that failed validation.
  // n_codec_errors is left untouched (still set on device by
  // decoder_status_reduce after nvcomp).
  //
  // Returns 0 on success (no parse errors); non-zero if any chunk failed
  // header validation. Callers should also inspect
  // out_totals->n_parse_errors and surface the wave as DAMACY_DECODE.
  int blosc1_host_parse(struct threadpool* pool,
                        const struct blosc1_host_chunk* chunks,
                        uint32_t n_chunks,
                        struct blosc1_host_scratch scratch,
                        struct blosc1_host_fanout zstd,
                        struct blosc1_host_fanout lz4,
                        struct gpu_memcpy_op* memcpy_ops,
                        struct gpu_shuffle_op* unshuffle_ops,
                        struct gpu_shuffle_op* bitunshuffle_ops,
                        struct blosc1_totals* out_totals);

#ifdef __cplusplus
}
#endif
