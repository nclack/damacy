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

  // Inputs + outputs for blosc1_host_parse. Designated-initializer
  // construction at the call site keeps the names visible:
  //
  //   blosc1_host_parse(&(struct blosc1_host_parse_args){
  //       .pool = compute_pool,
  //       .chunks = h_chunks, .n_chunks = n,
  //       .scratch = wave->scratch,
  //       .zstd = h_zstd_fan, .lz4 = h_lz4_fan,
  //       .memcpy_ops = h_memcpy_ops,
  //       .unshuffle_ops = h_unshuffle_ops,
  //       .bitunshuffle_ops = h_bitunshuffle_ops,
  //       .out_totals = h_blosc1_totals,
  //   });
  struct blosc1_host_parse_args
  {
    // Inputs.
    struct threadpool* pool; // NULL or nworkers==0 → serial on caller
    const struct blosc1_host_chunk* chunks;
    uint32_t n_chunks;

    // Per-wave scratch (allocated once at wave_init, reused).
    struct blosc1_host_scratch scratch;

    // Outputs filled in-place. Pointer slots in the SOAs hold device
    // addresses (chunks[i].d_compressed / d_decompressed + offset).
    struct blosc1_host_fanout zstd;
    struct blosc1_host_fanout lz4;
    struct gpu_memcpy_op* memcpy_ops;
    struct gpu_shuffle_op* unshuffle_ops;
    struct gpu_shuffle_op* bitunshuffle_ops;

    // Totals are fully written: count fields from the per-chunk scan
    // and n_parse_errors from the count of chunks that failed
    // validation. n_codec_errors is left untouched (still set on
    // device by decoder_status_reduce after nvcomp).
    struct blosc1_totals* out_totals;
  };

  // Parse args->n_chunks blosc1 chunks on args->pool, fill the fanout /
  // ops arrays, produce wave-level totals.
  //
  // Returns 0 on success (no parse errors); non-zero if any chunk
  // failed header validation. Callers should also inspect
  // args->out_totals->n_parse_errors and surface the wave as
  // DAMACY_DECODE.
  int blosc1_host_parse(const struct blosc1_host_parse_args* args);

#ifdef __cplusplus
}
#endif
