#include "decoder/blosc1_host.h"

#include "damacy_limits.h"
#include "log/log.h"
#include "threadpool/threadpool.h"
#include "util/prelude.h"
#include "zarr/zarr_chunk_layout.h"
#include "zarr/zarr_metadata.h"

#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

#define BLOSC1_HEADER_BYTES 16u

const char*
blosc1_host_parse_err_str(uint8_t err)
{
  switch (err) {
    case 0:
      return "ok";
    case 1:
      return "compressed_nbytes < 16";
    case 2:
      return "blocksize == 0";
    case 3:
      return "header.nbytes != decompressed_nbytes";
    case 4:
      return "header.cbytes != compressed_nbytes";
    case 5:
      return "nblocks > DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK";
    case 6:
      return "typesize out of bounds";
    case 7:
      return "header.compformat does not match codec_id";
    case 8:
      return "unsupported codec_id";
    case 9:
      return "bstart out of range";
    case 10:
      return "header.nbytes > DAMACY_BLOSC_MAX_CHUNK_UNCOMPRESSED_BYTES";
    case 11:
      return "truncated block cbytes prefix";
    case 12:
      return "chunk layout disagrees with array-level layout";
    default:
      return "unknown";
  }
}

static inline uint32_t
read_u32_le(const uint8_t* p)
{
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

static inline uint8_t
inner_codec_compformat(uint8_t codec)
{
  if (codec == CODEC_BLOSC_ZSTD)
    return 4;
  return 0;
}

// Compute block_ends[bi] = end-of-payload offset for block bi.
//
// blosc1's bstarts[] is in writer-completion order, not file order. To
// derive a block's extent we need the next bstart in ascending order
// (or h.cbytes for the last). We do this with an indirect insertion
// sort over `order[]` (n ≤ 32) and walk it once to populate ends.
static void
fill_block_ends(const uint32_t* bstarts,
                uint32_t* block_ends,
                uint32_t nblocks,
                uint32_t cbytes)
{
  // order[] = block indices sorted by (bstarts[i], i) ascending.
  uint32_t order[DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK];
  for (uint32_t i = 0; i < nblocks; ++i)
    order[i] = i;
  for (uint32_t i = 1; i < nblocks; ++i) {
    const uint32_t key_idx = order[i];
    const uint32_t key_off = bstarts[key_idx];
    uint32_t j = i;
    while (j > 0) {
      const uint32_t prev_idx = order[j - 1];
      const uint32_t prev_off = bstarts[prev_idx];
      if (prev_off < key_off || (prev_off == key_off && prev_idx < key_idx))
        break;
      order[j] = order[j - 1];
      --j;
    }
    order[j] = key_idx;
  }
  for (uint32_t r = 0; r < nblocks; ++r) {
    const uint32_t bi = order[r];
    block_ends[bi] = (r + 1u < nblocks) ? bstarts[order[r + 1u]] : cbytes;
  }
}

// Walks the per-block prefix chain for one CODEC_BLOSC_ZSTD chunk and
// fills counts. Caller has validated err == 0 and !memcpyed and has
// already populated bstarts[] / block_ends[]. blosc1-zstd has one
// substream per block. Sets h->err = 11 if any block's 4-byte cbytes
// prefix would read past block_ends[bi], or if the prefix value claims
// more bytes than remain in the block. On err=11 counts are left zero
// and emit_one skips the chunk via its h->err check.
static void
walk_count(const uint8_t* p,
           struct blosc1_chunk_hdr* h,
           const uint32_t* bstarts,
           const uint32_t* block_ends,
           struct blosc1_chunk_counts* c)
{
  const uint32_t per_stream_dst = h->blocksize;

  uint32_t n_codec = 0;
  uint32_t n_raw = 0;
  // end >= payload_lo >= 20 and cur <= end after the prefix read, so
  // the subtractions below can't underflow (cf. h.cbytes cap in
  // parse_count_one).
  for (uint32_t bi = 0; bi < h->nblocks; ++bi) {
    const uint32_t end = block_ends[bi];
    uint32_t cur = bstarts[bi];
    if (cur > end - 4u) {
      h->err = 11;
      return;
    }
    const uint32_t cb = read_u32_le(p + cur);
    cur += 4u;
    if (cb > end - cur) {
      h->err = 11;
      return;
    }
    if (cb == per_stream_dst)
      ++n_raw;
    else
      ++n_codec;
  }
  c->n_zstd = n_codec;
  c->n_memcpy = n_raw;
  c->n_unshuffle = h->shuffle;
  c->n_bitunshuffle = h->bitshuffle;
}

// Returns h.err (0 on success). Populates per-block bstarts / block_ends
// in scratch on the !memcpyed CODEC_BLOSC_* path so emit can reuse them.
static uint8_t
parse_count_one(const struct blosc1_host_chunk* in,
                struct blosc1_chunk_hdr* out_hdr,
                struct blosc1_chunk_counts* out_counts,
                uint32_t* bstarts_slot,
                uint32_t* block_ends_slot)
{
  struct blosc1_chunk_hdr h = { 0 };
  struct blosc1_chunk_counts c = { 0 };
  h.codec_id = in->codec_id;

  if (in->codec_id == CODEC_FILL)
    goto Done; // counts stay zero; assemble handles the broadcast
  if (in->codec_id == CODEC_NONE) {
    c.n_memcpy = 1;
    goto Done;
  }
  if (in->codec_id == CODEC_ZSTD) {
    c.n_zstd = 1;
    goto Done;
  }
  if (in->codec_id != CODEC_BLOSC_ZSTD) {
    h.err = 8;
    goto Done;
  }

  if (in->compressed_nbytes < BLOSC1_HEADER_BYTES) {
    h.err = 1;
    goto Done;
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
  if (h.nbytes > DAMACY_BLOSC_MAX_CHUNK_UNCOMPRESSED_BYTES) {
    h.err = 10;
    goto Done;
  }
  // Safe ceil-div: with nbytes capped above, the addition can't overflow
  // for any blocksize >= 1.
  h.nblocks = h.nbytes / h.blocksize + (h.nbytes % h.blocksize != 0u);
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
  // Defensive cap: the GPU shuffle / bitshuffle kernels compute
  // blocksize / typesize for thread-count sizing and would behave
  // pathologically (or divide by zero) for adversarial values. 8 covers
  // all source dtypes we currently accept (u8…f64); raise alongside
  // dtype support if/when that changes.
  if (h.typesize == 0 || h.typesize > 8u) {
    h.err = 6;
    goto Done;
  }
  if (h.compformat != inner_codec_compformat(in->codec_id)) {
    h.err = 7;
    goto Done;
  }

  // Array-level layout invariant: every chunk in a zarr array shares
  // typesize/blocksize/nblocks/shuffle. The planner cached the first
  // probed layout on the sample_plan; assert subsequent chunks agree
  // so wave_pool's pre-sized caps stay valid.
  if (in->layout) {
    const struct chunk_layout* L = in->layout;
    if (h.typesize != L->typesize || h.blocksize != L->blocksize ||
        h.nblocks != L->nblocks || h.shuffle != L->shuffle ||
        h.bitshuffle != L->bitshuffle || h.memcpyed != L->memcpyed ||
        ((h.compformat == 4) != (L->codec_id == CODEC_BLOSC_ZSTD))) {
      h.err = 12;
      goto Done;
    }
  }

  if (h.memcpyed) {
    c.n_memcpy = 1;
    goto Done;
  }

  const uint32_t payload_lo = BLOSC1_HEADER_BYTES + 4u * h.nblocks;
  if (payload_lo > h.cbytes) {
    h.err = 9;
    goto Done;
  }
  for (uint32_t bi = 0; bi < h.nblocks; ++bi) {
    const uint32_t bs = read_u32_le(p + BLOSC1_HEADER_BYTES + 4u * bi);
    if (bs < payload_lo || bs >= h.cbytes) {
      h.err = 9;
      goto Done;
    }
    bstarts_slot[bi] = bs;
  }
  fill_block_ends(bstarts_slot, block_ends_slot, h.nblocks, h.cbytes);
  walk_count(p, &h, bstarts_slot, block_ends_slot, &c);

Done:
  *out_hdr = h;
  *out_counts = c;
  return h.err;
}

static void
emit_one(const struct blosc1_host_chunk* in,
         const struct blosc1_chunk_hdr* h,
         const struct blosc1_chunk_offsets* o,
         const uint32_t* bstarts_slot,
         const uint32_t* block_ends_slot,
         struct blosc1_host_fanout zstd,
         struct gpu_memcpy_op* memcpy_ops,
         struct assemble_chunk* chunk_meta)
{
  if (h->err)
    return;

  uint8_t* d_decomp = (uint8_t*)in->d_decompressed;
  uint8_t* d_comp = (uint8_t*)in->d_compressed;

  // Defaults: assemble reads bytes directly; flipped below if the
  // chunk's blosc header declares a shuffle.
  chunk_meta->shuffle_mode = ASSEMBLE_SHUFFLE_NONE;
  chunk_meta->shuffle_typesize = 0;
  chunk_meta->shuffle_blocksize = 0;

  if (in->codec_id == CODEC_FILL)
    return; // no codec / memcpy work; assemble broadcasts fill_value
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

  if (h->memcpyed) {
    const uint32_t overhead = BLOSC1_HEADER_BYTES + 4u * h->nblocks;
    struct gpu_memcpy_op* slot = &memcpy_ops[o->memcpy_off];
    slot->d_src = d_comp + overhead;
    slot->d_dst = d_decomp;
    slot->nbytes = in->decompressed_nbytes;
    return;
  }

  const uint8_t* p = in->h_compressed;
  const uint32_t per_stream_dst = h->blocksize;
  const uint32_t codec_base = o->zstd_off;
  const uint32_t memcpy_base = o->memcpy_off;

  uint32_t k_codec = 0;
  uint32_t k_raw = 0;
  // See walk_count: end and cur are bounded so the subtractions below
  // can't underflow. blosc1-zstd has one substream per block.
  for (uint32_t bi = 0; bi < h->nblocks; ++bi) {
    const uint32_t end = block_ends_slot[bi];
    const uint32_t block_dst_off = bi * h->blocksize;
    uint32_t cur = bstarts_slot[bi];
    if (cur > end - 4u)
      continue;
    const uint32_t cb = read_u32_le(p + cur);
    cur += 4u;
    if (cb > end - cur)
      continue;
    void* dst = d_decomp + block_dst_off;
    if (cb == per_stream_dst) {
      struct gpu_memcpy_op* slot = &memcpy_ops[memcpy_base + k_raw];
      slot->d_src = d_comp + cur;
      slot->d_dst = dst;
      slot->nbytes = per_stream_dst;
      ++k_raw;
    } else {
      const uint32_t idx = codec_base + k_codec;
      zstd.comp_ptrs[idx] = d_comp + cur;
      zstd.comp_sizes[idx] = cb;
      zstd.decomp_ptrs[idx] = dst;
      zstd.decomp_buf_sizes[idx] = per_stream_dst;
      ++k_codec;
    }
  }

  if (h->shuffle) {
    chunk_meta->shuffle_mode = ASSEMBLE_SHUFFLE_BYTE;
    chunk_meta->shuffle_typesize = h->typesize;
    chunk_meta->shuffle_blocksize = h->blocksize;
  } else if (h->bitshuffle) {
    chunk_meta->shuffle_mode = ASSEMBLE_SHUFFLE_BIT;
    chunk_meta->shuffle_typesize = h->typesize;
    chunk_meta->shuffle_blocksize = h->blocksize;
  }
  (void)o;
}

struct parse_ctx
{
  const struct blosc1_host_chunk* chunks;
  struct blosc1_chunk_hdr* hdrs;
  struct blosc1_chunk_counts* counts;
  const struct blosc1_chunk_offsets* offsets;
  uint32_t* bstarts;    // cap * DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK
  uint32_t* block_ends; // cap * DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK
  struct blosc1_host_fanout zstd;
  struct gpu_memcpy_op* memcpy_ops;
  struct assemble_chunk* assemble_chunks;
  _Atomic uint32_t n_parse_errors;
};

static inline uint32_t*
chunk_slot(uint32_t* base, size_t i)
{
  return base + i * DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK;
}

static void
phase_count(size_t i, int tid, void* vctx)
{
  (void)tid;
  struct parse_ctx* ctx = (struct parse_ctx*)vctx;
  uint8_t err = parse_count_one(&ctx->chunks[i],
                                &ctx->hdrs[i],
                                &ctx->counts[i],
                                chunk_slot(ctx->bstarts, i),
                                chunk_slot(ctx->block_ends, i));
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
           chunk_slot(ctx->bstarts, i),
           chunk_slot(ctx->block_ends, i),
           ctx->zstd,
           ctx->memcpy_ops,
           &ctx->assemble_chunks[i]);
}

static void
scan_offsets(const struct blosc1_chunk_counts* counts,
             struct blosc1_chunk_offsets* offsets,
             uint32_t n_chunks,
             struct blosc1_totals* totals)
{
  uint32_t z = 0, m = 0, u = 0, b = 0;
  for (uint32_t i = 0; i < n_chunks; ++i) {
    offsets[i].zstd_off = z;
    offsets[i].memcpy_off = m;
    offsets[i].unshuffle_off = u;
    offsets[i].bitunshuffle_off = b;
    z += counts[i].n_zstd;
    m += counts[i].n_memcpy;
    u += counts[i].n_unshuffle;
    b += counts[i].n_bitunshuffle;
  }
  totals->n_zstd = z;
  totals->n_memcpy = m;
  totals->n_unshuffle = u;
  totals->n_bitunshuffle = b;
}

int
blosc1_host_parse(const struct blosc1_host_parse_args* args)
{
  CHECK(Fail, args);
  CHECK(Fail, args->out_totals);
  struct blosc1_totals* out_totals = args->out_totals;

  // n_codec_errors is owned by decoder_status_reduce; don't touch.
  out_totals->n_zstd = 0;
  out_totals->n_memcpy = 0;
  out_totals->n_unshuffle = 0;
  out_totals->n_bitunshuffle = 0;
  out_totals->n_parse_errors = 0;
  if (args->n_chunks == 0)
    return 0;

  CHECK(Fail, args->chunks);
  CHECK(Fail, args->scratch.hdrs);
  CHECK(Fail, args->scratch.counts);
  CHECK(Fail, args->scratch.offsets);
  CHECK(Fail, args->scratch.bstarts);
  CHECK(Fail, args->scratch.block_ends);
  CHECK(Fail, args->zstd.comp_ptrs);
  CHECK(Fail, args->zstd.comp_sizes);
  CHECK(Fail, args->zstd.decomp_ptrs);
  CHECK(Fail, args->zstd.decomp_buf_sizes);
  CHECK(Fail, args->memcpy_ops);
  CHECK(Fail, args->assemble_chunks);

  struct parse_ctx ctx = {
    .chunks = args->chunks,
    .hdrs = args->scratch.hdrs,
    .counts = args->scratch.counts,
    .offsets = args->scratch.offsets,
    .bstarts = args->scratch.bstarts,
    .block_ends = args->scratch.block_ends,
    .zstd = args->zstd,
    .memcpy_ops = args->memcpy_ops,
    .assemble_chunks = args->assemble_chunks,
  };
  atomic_store_explicit(&ctx.n_parse_errors, 0u, memory_order_relaxed);

  threadpool_for_n_dynamic(args->pool, args->n_chunks, phase_count, &ctx);

  out_totals->n_parse_errors =
    atomic_load_explicit(&ctx.n_parse_errors, memory_order_relaxed);
  if (out_totals->n_parse_errors > 0) {
    log_error("blosc1_host_parse: %u chunk(s) failed parse",
              out_totals->n_parse_errors);
    return 1;
  }

  scan_offsets(
    args->scratch.counts, args->scratch.offsets, args->n_chunks, out_totals);

  threadpool_for_n_dynamic(args->pool, args->n_chunks, phase_emit, &ctx);
  return 0;
Fail:
  return 1;
}
