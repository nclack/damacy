#include "decoder/blosc1_host.h"

#include "damacy_limits.h"
#include "log/log.h"
#include "threadpool/threadpool.h"
#include "zarr/zarr_metadata.h"

#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

// blosc1 chunk header: 16 bytes at offset 0; bstarts[nblocks] follows.
#define BLOSC1_HEADER_BYTES 16u

static inline uint32_t
read_u32_le(const uint8_t* p)
{
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

static inline uint8_t
inner_codec_compformat(uint8_t codec)
{
  if (codec == CODEC_BLOSC_LZ4)
    return 1;
  if (codec == CODEC_BLOSC_ZSTD)
    return 4;
  return 0;
}

// Rank-sort bstarts into ascending order; ties broken by block index.
// nblocks <= DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK (currently 64), so an
// O(n²) rank-sort is comfortably under a microsecond per chunk.
static void
sort_bstarts(const uint32_t* bstarts, uint32_t* sorted, uint32_t nblocks)
{
  for (uint32_t bi = 0; bi < nblocks; ++bi) {
    uint32_t r = 0;
    for (uint32_t j = 0; j < nblocks; ++j) {
      const uint32_t a = bstarts[j];
      const uint32_t b = bstarts[bi];
      if (a < b || (a == b && j < bi))
        ++r;
    }
    sorted[r] = bstarts[bi];
  }
}

// Walk one CODEC_BLOSC_* chunk's bstarts + per-substream prefixes,
// filling `*c` with substream counts. Returns the chunk's per-substream
// nstreams (== typesize for LZ4 multi-stream, else 1) for emit reuse.
// Caller has already validated h->err == 0 and !h->memcpyed.
static uint32_t
walk_count(const uint8_t* p,
           const struct blosc1_chunk_hdr* h,
           uint8_t codec,
           const uint32_t* bstarts,
           const uint32_t* sorted,
           struct blosc1_chunk_counts* c)
{
  const uint32_t nstreams =
    (codec == CODEC_BLOSC_LZ4 && h->typesize > 1) ? (uint32_t)h->typesize : 1u;
  const uint32_t per_stream_dst =
    (nstreams == 1u) ? h->blocksize : (h->blocksize / (uint32_t)h->typesize);

  uint32_t n_codec = 0;
  uint32_t n_raw = 0;
  for (uint32_t bi = 0; bi < h->nblocks; ++bi) {
    // Block bi's payload extent ends at the next bstart in sorted order
    // (or at cbytes for the last block).
    uint32_t r = 0;
    for (uint32_t j = 0; j < h->nblocks; ++j) {
      const uint32_t a = bstarts[j];
      const uint32_t b = bstarts[bi];
      if (a < b || (a == b && j < bi))
        ++r;
    }
    const uint32_t end = (r + 1u < h->nblocks) ? sorted[r + 1u] : h->cbytes;
    uint32_t cur = bstarts[bi];
    for (uint32_t k = 0; k < nstreams; ++k) {
      if (cur + 4u > end)
        break;
      const uint32_t cb = read_u32_le(p + cur);
      cur += 4u;
      if (cur + cb > end)
        break;
      if (cb == per_stream_dst)
        ++n_raw;
      else
        ++n_codec;
      cur += cb;
    }
  }
  if (codec == CODEC_BLOSC_LZ4)
    c->n_lz4 = n_codec;
  else
    c->n_zstd = n_codec;
  c->n_memcpy = n_raw;
  c->n_unshuffle = h->shuffle;
  c->n_bitunshuffle = h->bitshuffle;
  return per_stream_dst;
}

// One chunk's parse + count. Sets hdrs[i] and counts[i]; returns the
// chunk's err (0 on success).
static uint8_t
parse_count_one(const struct blosc1_host_chunk* in,
                struct blosc1_chunk_hdr* out_hdr,
                struct blosc1_chunk_counts* out_counts)
{
  struct blosc1_chunk_hdr h = { 0 };
  struct blosc1_chunk_counts c = { 0 };
  h.codec_id = in->codec_id;

  if (in->codec_id == CODEC_NONE) {
    c.n_memcpy = 1;
    *out_hdr = h;
    *out_counts = c;
    return 0;
  }
  if (in->codec_id == CODEC_ZSTD) {
    c.n_zstd = 1;
    *out_hdr = h;
    *out_counts = c;
    return 0;
  }
  if (in->codec_id != CODEC_BLOSC_LZ4 && in->codec_id != CODEC_BLOSC_ZSTD) {
    h.err = 8;
    *out_hdr = h;
    *out_counts = c;
    return h.err;
  }

  if (in->compressed_nbytes < BLOSC1_HEADER_BYTES) {
    h.err = 1;
    *out_hdr = h;
    *out_counts = c;
    return h.err;
  }
  const uint8_t* p = in->h_compressed;
  const uint8_t flags = p[2];
  h.typesize = p[3];
  h.shuffle = flags & 0x01u;
  h.memcpyed = (flags >> 1) & 0x01u;
  h.bitshuffle = (flags >> 2) & 0x01u;
  h.compformat = (flags >> 5) & 0x07u;
  h.nbytes = read_u32_le(p + 4);
  h.blocksize = read_u32_le(p + 8);
  h.cbytes = read_u32_le(p + 12);
  if (h.blocksize == 0) {
    h.err = 2;
    goto Done;
  }
  h.nblocks = (h.nbytes + h.blocksize - 1) / h.blocksize;
  if (h.nbytes != in->decompressed_nbytes) {
    h.err = 3;
    goto Done;
  }
  if (h.cbytes != in->compressed_nbytes) {
    h.err = 4;
    goto Done;
  }
  if (h.nblocks > DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK) {
    h.err = 5;
    goto Done;
  }
  if (h.typesize == 0 || h.typesize > DAMACY_BLOSC_MAX_TYPESIZE) {
    h.err = 6;
    goto Done;
  }
  if (h.compformat != inner_codec_compformat(in->codec_id)) {
    h.err = 7;
    goto Done;
  }

  if (h.memcpyed) {
    c.n_memcpy = 1;
    goto Done;
  }

  uint32_t bstarts[DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK] = { 0 };
  uint32_t sorted[DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK] = { 0 };
  for (uint32_t bi = 0; bi < h.nblocks; ++bi)
    bstarts[bi] = read_u32_le(p + BLOSC1_HEADER_BYTES + 4u * bi);
  sort_bstarts(bstarts, sorted, h.nblocks);
  walk_count(p, &h, in->codec_id, bstarts, sorted, &c);

Done:
  *out_hdr = h;
  *out_counts = c;
  return h.err;
}

// Emit one chunk's fanout / op slots. `o` is the chunk's offsets into
// each output array, computed by the scan pass.
static void
emit_one(const struct blosc1_host_chunk* in,
         const struct blosc1_chunk_hdr* h,
         const struct blosc1_chunk_offsets* o,
         struct blosc1_host_fanout zstd,
         struct blosc1_host_fanout lz4,
         struct gpu_memcpy_op* memcpy_ops,
         struct gpu_shuffle_op* unshuffle_ops,
         struct gpu_shuffle_op* bitunshuffle_ops)
{
  if (h->err)
    return;

  uint8_t* d_decomp = (uint8_t*)in->d_decompressed;
  uint8_t* d_comp = (uint8_t*)in->d_compressed;

  if (in->codec_id == CODEC_NONE) {
    struct gpu_memcpy_op* slot = &memcpy_ops[o->memcpy_off];
    slot->d_src = d_comp;
    slot->d_dst = d_decomp;
    slot->nbytes = in->decompressed_nbytes;
    return;
  }
  if (in->codec_id == CODEC_ZSTD) {
    zstd.comp_ptrs[o->zstd_off] = d_comp;
    zstd.comp_sizes[o->zstd_off] = in->compressed_nbytes;
    zstd.decomp_ptrs[o->zstd_off] = d_decomp;
    zstd.decomp_buf_sizes[o->zstd_off] = in->decompressed_nbytes;
    return;
  }

  // CODEC_BLOSC_*
  if (h->memcpyed) {
    const uint32_t overhead = BLOSC1_HEADER_BYTES + 4u * h->nblocks;
    struct gpu_memcpy_op* slot = &memcpy_ops[o->memcpy_off];
    slot->d_src = d_comp + overhead;
    slot->d_dst = d_decomp;
    slot->nbytes = in->decompressed_nbytes;
    return;
  }

  uint32_t bstarts[DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK] = { 0 };
  uint32_t sorted[DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK] = { 0 };
  const uint8_t* p = in->h_compressed;
  for (uint32_t bi = 0; bi < h->nblocks; ++bi)
    bstarts[bi] = read_u32_le(p + BLOSC1_HEADER_BYTES + 4u * bi);
  sort_bstarts(bstarts, sorted, h->nblocks);

  const uint8_t codec = in->codec_id;
  const uint32_t nstreams =
    (codec == CODEC_BLOSC_LZ4 && h->typesize > 1) ? (uint32_t)h->typesize : 1u;
  const uint32_t per_stream_dst =
    (nstreams == 1u) ? h->blocksize : (h->blocksize / (uint32_t)h->typesize);
  struct blosc1_host_fanout fan = (codec == CODEC_BLOSC_LZ4) ? lz4 : zstd;
  const uint32_t codec_base =
    (codec == CODEC_BLOSC_LZ4) ? o->lz4_off : o->zstd_off;
  const uint32_t memcpy_base = o->memcpy_off;

  uint32_t k_codec = 0;
  uint32_t k_raw = 0;
  for (uint32_t bi = 0; bi < h->nblocks; ++bi) {
    uint32_t r = 0;
    for (uint32_t j = 0; j < h->nblocks; ++j) {
      const uint32_t a = bstarts[j];
      const uint32_t b = bstarts[bi];
      if (a < b || (a == b && j < bi))
        ++r;
    }
    const uint32_t end = (r + 1u < h->nblocks) ? sorted[r + 1u] : h->cbytes;
    const uint32_t block_dst_off = bi * h->blocksize;
    uint32_t cur = bstarts[bi];
    for (uint32_t k = 0; k < nstreams; ++k) {
      if (cur + 4u > end)
        break;
      const uint32_t cb = read_u32_le(p + cur);
      cur += 4u;
      if (cur + cb > end)
        break;
      void* dst = d_decomp + block_dst_off + k * per_stream_dst;
      if (cb == per_stream_dst) {
        struct gpu_memcpy_op* slot = &memcpy_ops[memcpy_base + k_raw];
        slot->d_src = d_comp + cur;
        slot->d_dst = dst;
        slot->nbytes = per_stream_dst;
        ++k_raw;
      } else {
        const uint32_t idx = codec_base + k_codec;
        fan.comp_ptrs[idx] = d_comp + cur;
        fan.comp_sizes[idx] = cb;
        fan.decomp_ptrs[idx] = dst;
        fan.decomp_buf_sizes[idx] = per_stream_dst;
        ++k_codec;
      }
      cur += cb;
    }
  }

  if (h->shuffle) {
    struct gpu_shuffle_op* slot = &unshuffle_ops[o->unshuffle_off];
    slot->d_buf = d_decomp;
    slot->blocksize = h->blocksize;
    slot->typesize = h->typesize;
    slot->nblocks_full = h->nblocks;
    slot->tail_nbytes = 0;
  }
  if (h->bitshuffle) {
    struct gpu_shuffle_op* slot = &bitunshuffle_ops[o->bitunshuffle_off];
    slot->d_buf = d_decomp;
    slot->blocksize = h->blocksize;
    slot->typesize = h->typesize;
    slot->nblocks_full = h->nblocks;
    slot->tail_nbytes = 0;
  }
}

// Three-phase parallel-for driver state. Phase A and Phase C share a
// pointer to this; Phase B (the scan) runs serially on the caller.
struct parse_ctx
{
  const struct blosc1_host_chunk* chunks;
  struct blosc1_chunk_hdr* hdrs;
  struct blosc1_chunk_counts* counts;
  const struct blosc1_chunk_offsets* offsets;
  struct blosc1_host_fanout zstd;
  struct blosc1_host_fanout lz4;
  struct gpu_memcpy_op* memcpy_ops;
  struct gpu_shuffle_op* unshuffle_ops;
  struct gpu_shuffle_op* bitunshuffle_ops;
  _Atomic uint32_t n_parse_errors;
};

static void
phase_count(size_t i, int tid, void* vctx)
{
  (void)tid;
  struct parse_ctx* ctx = (struct parse_ctx*)vctx;
  uint8_t err =
    parse_count_one(&ctx->chunks[i], &ctx->hdrs[i], &ctx->counts[i]);
  if (err)
    atomic_fetch_add_explicit(&ctx->n_parse_errors, 1u, memory_order_relaxed);
}

static void
phase_emit(size_t i, int tid, void* vctx)
{
  (void)tid;
  struct parse_ctx* ctx = (struct parse_ctx*)vctx;
  emit_one(&ctx->chunks[i],
           &ctx->hdrs[i],
           &ctx->offsets[i],
           ctx->zstd,
           ctx->lz4,
           ctx->memcpy_ops,
           ctx->unshuffle_ops,
           ctx->bitunshuffle_ops);
}

// Serial exclusive scan over the five count fields, simultaneously
// producing wave totals. n_chunks <= DAMACY_MAX_CHUNKS_PER_WAVE so the
// loop's bounded; not worth parallelising vs. the dispatch overhead.
static void
scan_offsets(const struct blosc1_chunk_counts* counts,
             struct blosc1_chunk_offsets* offsets,
             uint32_t n_chunks,
             struct blosc1_totals* totals)
{
  uint32_t z = 0, l = 0, m = 0, u = 0, b = 0;
  for (uint32_t i = 0; i < n_chunks; ++i) {
    offsets[i].zstd_off = z;
    offsets[i].lz4_off = l;
    offsets[i].memcpy_off = m;
    offsets[i].unshuffle_off = u;
    offsets[i].bitunshuffle_off = b;
    z += counts[i].n_zstd;
    l += counts[i].n_lz4;
    m += counts[i].n_memcpy;
    u += counts[i].n_unshuffle;
    b += counts[i].n_bitunshuffle;
  }
  totals->n_zstd = z;
  totals->n_lz4 = l;
  totals->n_memcpy = m;
  totals->n_unshuffle = u;
  totals->n_bitunshuffle = b;
}

int
blosc1_host_parse(struct threadpool* pool,
                  const struct blosc1_host_chunk* chunks,
                  uint32_t n_chunks,
                  struct blosc1_host_scratch scratch,
                  struct blosc1_host_fanout zstd,
                  struct blosc1_host_fanout lz4,
                  struct gpu_memcpy_op* memcpy_ops,
                  struct gpu_shuffle_op* unshuffle_ops,
                  struct gpu_shuffle_op* bitunshuffle_ops,
                  struct blosc1_totals* out_totals)
{
  // n_codec_errors is set by decoder_status_reduce after nvcomp; leave
  // it alone here. Other fields are fully written below.
  out_totals->n_zstd = 0;
  out_totals->n_lz4 = 0;
  out_totals->n_memcpy = 0;
  out_totals->n_unshuffle = 0;
  out_totals->n_bitunshuffle = 0;
  out_totals->n_parse_errors = 0;
  if (n_chunks == 0)
    return 0;

  struct parse_ctx ctx = {
    .chunks = chunks,
    .hdrs = scratch.hdrs,
    .counts = scratch.counts,
    .offsets = scratch.offsets,
    .zstd = zstd,
    .lz4 = lz4,
    .memcpy_ops = memcpy_ops,
    .unshuffle_ops = unshuffle_ops,
    .bitunshuffle_ops = bitunshuffle_ops,
  };
  atomic_store_explicit(&ctx.n_parse_errors, 0u, memory_order_relaxed);

  threadpool_for_n_dynamic(pool, n_chunks, phase_count, &ctx);

  out_totals->n_parse_errors =
    atomic_load_explicit(&ctx.n_parse_errors, memory_order_relaxed);
  if (out_totals->n_parse_errors > 0) {
    log_error("blosc1_host_parse: %u chunk(s) failed parse",
              out_totals->n_parse_errors);
    return 1;
  }

  scan_offsets(scratch.counts, scratch.offsets, n_chunks, out_totals);

  threadpool_for_n_dynamic(pool, n_chunks, phase_emit, &ctx);
  return 0;
}
