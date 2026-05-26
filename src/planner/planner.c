#include "planner/planner.h"

#include "damacy_config.h"
#include "dtype/dtype.h"
#include "log/log.h"
#include "planner/coalesce.h"
#include "planner/group_chunks.h"
#include "prefetch/prefetch_cache.h"
#include "prefetch/shard_index.h"
#include "util/path_intern.h"
#include "util/prelude.h"
#include "util/strbuf.h"
#include "zarr/sample_shard_iterator.h"
#include "zarr/zarr_metadata.h"
#include "zarr/zarr_shard_index.h"

#include <stdlib.h>
#include <string.h>

struct planner
{
  struct planner_config cfg;
  struct strbuf path_sb;
  uint32_t* scratch_u32;
  uint32_t scratch_u32_cap;
  struct read_op* scratch_ops;
  uint32_t scratch_ops_cap;
  struct chunk_plan* scratch_chunk_plans;
  uint32_t scratch_chunk_plans_cap;
};

// `need` = pre-coalesce n_read_ops (== n_chunk_plans). The uint32
// buffer is sized to 3*need; post-coalesce n_read_ops <= need so
// group_chunks_by_read's n_read_ops+1 slots fit in the same buffer.
static enum damacy_status
planner_ensure_scratch(struct planner* self, uint32_t need)
{
  if (need > self->scratch_u32_cap) {
    free(self->scratch_u32);
    self->scratch_u32 = NULL;
    self->scratch_u32_cap = 0;
    uint32_t* mem = (uint32_t*)malloc((size_t)need * 3u * sizeof(uint32_t));
    if (!mem)
      return DAMACY_OOM;
    self->scratch_u32 = mem;
    self->scratch_u32_cap = need;
  }
  if (need > self->scratch_ops_cap) {
    free(self->scratch_ops);
    self->scratch_ops = NULL;
    self->scratch_ops_cap = 0;
    struct read_op* mem =
      (struct read_op*)malloc((size_t)need * sizeof(struct read_op));
    if (!mem)
      return DAMACY_OOM;
    self->scratch_ops = mem;
    self->scratch_ops_cap = need;
  }
  if (need > self->scratch_chunk_plans_cap) {
    free(self->scratch_chunk_plans);
    self->scratch_chunk_plans = NULL;
    self->scratch_chunk_plans_cap = 0;
    struct chunk_plan* mem =
      (struct chunk_plan*)malloc((size_t)need * sizeof(struct chunk_plan));
    if (!mem)
      return DAMACY_OOM;
    self->scratch_chunk_plans = mem;
    self->scratch_chunk_plans_cap = need;
  }
  return DAMACY_OK;
}

// --- math helpers --------------------------------------------------------

// chunk_beg/chunk_end are inner-chunk-grid units (half-open).
static int
chunk_range(const struct damacy_aabb* aabb,
            const struct zarr_metadata* meta,
            uint64_t* chunk_beg,
            uint64_t* chunk_end)
{
  if (aabb->rank != meta->rank)
    return 1;
  for (uint8_t d = 0; d < meta->rank; ++d) {
    int64_t beg = aabb->dims[d].beg;
    int64_t end = aabb->dims[d].end;
    if (beg < 0 || end <= beg)
      return 1;
    if ((uint64_t)end > meta->shape[d])
      return 1;
    uint64_t chunk_extent = meta->inner_chunk_shape[d];
    if (chunk_extent == 0)
      return 1;
    chunk_beg[d] = (uint64_t)beg / chunk_extent;
    chunk_end[d] = ((uint64_t)end - 1) / chunk_extent + 1;
  }
  return 0;
}

// Row-major linear index into a multi-d grid given per-dim extents.
static uint64_t
row_major_linear(const uint64_t* coord, const uint64_t* extents, uint8_t rank)
{
  uint64_t linear_idx = 0;
  for (uint8_t d = 0; d < rank; ++d)
    linear_idx = linear_idx * extents[d] + coord[d];
  return linear_idx;
}

// Element strides (row-major) into a row-major chunk of shape `shape`.
static void
row_major_strides(const uint64_t* shape, uint8_t rank, int64_t* out_strides)
{
  if (rank == 0)
    return;
  out_strides[rank - 1] = 1;
  for (int d = (int)rank - 2; d >= 0; --d)
    out_strides[d] = out_strides[d + 1] * (int64_t)shape[d + 1];
}

// Bytes per inner chunk (uncompressed). Returns 0 on overflow during
// the product or if the result exceeds DAMACY_MAX_CHUNK_BYTES.
static uint64_t
inner_chunk_bytes(const struct zarr_metadata* meta)
{
  uint64_t total_bytes = dtype_bpe(meta->dtype);
  for (uint8_t d = 0; d < meta->rank; ++d) {
    uint64_t chunk_extent = meta->inner_chunk_shape[d];
    if (chunk_extent != 0 && total_bytes > UINT64_MAX / chunk_extent)
      return 0;
    total_bytes *= chunk_extent;
  }
  if (total_bytes > DAMACY_MAX_CHUNK_BYTES)
    return 0;
  return total_bytes;
}

static uint64_t
align_down_u64(uint64_t value, uint64_t alignment)
{
  return (value / alignment) * alignment;
}

static uint64_t
align_up_u64(uint64_t value, uint64_t alignment)
{
  return ((value + alignment - 1) / alignment) * alignment;
}

// --- public API ----------------------------------------------------------

enum damacy_status
planner_create(const struct planner_config* cfg, struct planner** out)
{
  struct planner* self = NULL;
  enum damacy_status status = DAMACY_INVAL;

  CHECK_SILENT(Error, out);
  *out = NULL;
  CHECK_SILENT(Error, cfg);
  CHECK_SILENT(Error, cfg->array_meta_cache);
  CHECK_SILENT(Error, cfg->chunk_layout_cache);
  CHECK_SILENT(Error, cfg->shard_index_cache);
  CHECK_SILENT(Error, cfg->page_alignment > 0);
  CHECK_SILENT(Error, cfg->max_chunks_per_wave > 0);
  CHECK_SILENT(Error, cfg->max_substreams_per_chunk > 0);

  status = DAMACY_OOM;
  self = (struct planner*)calloc(1, sizeof(*self));
  CHECK(Error, self);

  // Designated init zeroes any fields not explicitly named, including
  // path_sb (zero-init is documented as safe for struct strbuf).
  *self = (struct planner){ .cfg = *cfg };
  *out = self;
  return DAMACY_OK;

Error:
  planner_destroy(self);
  return status;
}

void
planner_destroy(struct planner* self)
{
  if (!self)
    return;
  strbuf_free(&self->path_sb);
  free(self->scratch_u32);
  free(self->scratch_ops);
  free(self->scratch_chunk_plans);
  free(self);
}

// Per-sample/per-shard invariants for a run of emit_chunk calls. The
// per-sample fields stay constant while iterating chunks; per-shard
// fields are refreshed when the iterator crosses into a new shard.
//
// shard_missing == 1 short-circuits the per-shard entry lookup and emits
// fill chunks for every chunk inside the shard (shard file absent).
struct emit_ctx
{
  // per-sample
  const struct planner_sample* sample;
  const struct zarr_metadata* meta;
  const uint64_t* inner_per_shard_dim; // [meta->rank]
  const uint64_t* chunk_beg;           // [meta->rank], first chunk for sample
  uint32_t sample_idx_in_batch;
  uint16_t batch_pool_slot;
  uint8_t codec_id;
  uint32_t decompressed_n_bytes; // == inner_chunk_bytes(meta) (validated)
  uint64_t page_alignment_bytes;
  // Mutable handle to this sample's plan entry. emit_chunk lazily
  // populates ->layout / ->layout_probed on the first non-fill emit.
  struct sample_plan* sp;
  const struct chunk_layout* chunk_layout;
  uint32_t max_substreams_per_chunk;
  // per-shard
  const struct zarr_shard_entry* shard_entries;
  uint64_t n_shard_entries;
  const char* interned_path;
  int shard_missing;
};

// Append a fill-mode chunk_plan + matching dummy read_op so the
// batch-level read_ops / chunk_plans arrays stay 1:1 with the
// chunk_plan stream that downstream peel walks in lockstep.
static enum damacy_status
emit_fill_chunk(const struct emit_ctx* ctx,
                const uint64_t* chunk_coord,
                struct planner_output* out)
{
  if (out->n_read_ops >= out->read_ops_cap ||
      out->n_chunk_plans >= out->chunk_plans_cap)
    return DAMACY_OOM;
  const struct zarr_metadata* meta = ctx->meta;

  uint32_t read_op_idx = out->n_read_ops;
  struct read_op* r = &out->read_ops[read_op_idx];
  *r = (struct read_op){ 0 };
  out->n_read_ops++;

  struct chunk_plan* cp = &out->chunk_plans[out->n_chunk_plans];
  *cp = (struct chunk_plan){
    .read_op_idx = read_op_idx,
    .offset_in_read = 0,
    .compressed_nbytes = 0,
    .decompressed_nbytes = ctx->decompressed_n_bytes,
    .batch_pool_slot = ctx->batch_pool_slot,
    .sample_idx_in_batch = (uint16_t)ctx->sample_idx_in_batch,
    .codec_id = (uint8_t)CODEC_FILL,
    .is_fill = 1,
  };
  for (uint8_t d = 0; d < meta->rank; ++d)
    cp->chunk_d[d] = (uint32_t)(chunk_coord[d] - ctx->chunk_beg[d]);
  out->n_chunk_plans++;
  return DAMACY_OK;
}

// Process one chunk: build read_op + chunk_plan, append to output.
// Empty shard entries and missing shard files emit a fill-mode chunk_plan
// referencing the array's fill_value rather than failing.
static enum damacy_status
emit_chunk(const struct emit_ctx* ctx,
           const uint64_t* chunk_coord,
           const uint64_t* local_inner,
           struct planner_output* out)
{
  const struct zarr_metadata* meta = ctx->meta;

  // Shard file absent: every chunk in the shard is fill.
  if (ctx->shard_missing)
    return emit_fill_chunk(ctx, chunk_coord, out);

  uint64_t entry_idx =
    row_major_linear(local_inner, ctx->inner_per_shard_dim, meta->rank);
  if (entry_idx >= ctx->n_shard_entries)
    return DAMACY_DECODE;

  const struct zarr_shard_entry* entry = &ctx->shard_entries[entry_idx];
  int off_sentinel = entry->offset == ZARR_SHARD_EMPTY_OFFSET;
  int nb_sentinel = entry->nbytes == ZARR_SHARD_EMPTY_NBYTES;
  if (off_sentinel && nb_sentinel)
    return emit_fill_chunk(ctx, chunk_coord, out);
  if (off_sentinel != nb_sentinel) {
    log_error("zarr shard index entry %llu half-sentinel "
              "(offset=0x%llx, nbytes=0x%llx); treating as corrupt",
              (unsigned long long)entry_idx,
              (unsigned long long)entry->offset,
              (unsigned long long)entry->nbytes);
    return DAMACY_DECODE;
  }

  if (entry->nbytes > DAMACY_MAX_CHUNK_BYTES)
    return DAMACY_DECODE;

  // Layout was pre-fetched by the prefetcher into chunk_layout_cache.
  // NULL means the array isn't blosc1 (decoder uses caps); the wave-
  // eligibility gate keeps layout_probed=0 chunks out until probed.
  if (ctx->sp && !ctx->sp->layout_probed && ctx->chunk_layout) {
    ctx->sp->layout = *ctx->chunk_layout;
    ctx->sp->layout_probed = 1;
  }

  // Page-aligned read window enclosing [offset, offset + nbytes).
  uint64_t page_alignment_bytes = ctx->page_alignment_bytes;
  uint64_t aligned_file_offset =
    align_down_u64(entry->offset, page_alignment_bytes);
  uint64_t aligned_end =
    align_up_u64(entry->offset + entry->nbytes, page_alignment_bytes);
  uint64_t read_n_bytes = aligned_end - aligned_file_offset;
  if (read_n_bytes > DAMACY_MAX_CHUNK_BYTES)
    return DAMACY_DECODE;
  uint32_t chunk_offset_in_read =
    (uint32_t)(entry->offset - aligned_file_offset);

  if (out->n_read_ops >= out->read_ops_cap ||
      out->n_chunk_plans >= out->chunk_plans_cap)
    return DAMACY_OOM;

  uint32_t read_op_idx = out->n_read_ops;
  struct read_op* r = &out->read_ops[read_op_idx];
  r->shard_path = ctx->interned_path;
  r->file_offset = aligned_file_offset;
  r->nbytes = (uint32_t)read_n_bytes;
  out->n_read_ops++;

  struct chunk_plan* cp = &out->chunk_plans[out->n_chunk_plans];
  *cp = (struct chunk_plan){
    .read_op_idx = read_op_idx,
    .offset_in_read = chunk_offset_in_read,
    .compressed_nbytes = (uint32_t)entry->nbytes,
    .decompressed_nbytes = ctx->decompressed_n_bytes,
    .batch_pool_slot = ctx->batch_pool_slot,
    .sample_idx_in_batch = (uint16_t)ctx->sample_idx_in_batch,
    .codec_id = ctx->codec_id,
  };
  for (uint8_t d = 0; d < meta->rank; ++d)
    cp->chunk_d[d] = (uint32_t)(chunk_coord[d] - ctx->chunk_beg[d]);

  out->n_chunk_plans++;
  return DAMACY_OK;
}

enum damacy_status
planner_plan(struct planner* self,
             const struct planner_sample* samples,
             uint32_t n_samples,
             uint16_t batch_pool_slot,
             const int64_t* dst_strides,
             uint8_t dst_full_rank,
             struct planner_output* out)
{
  enum damacy_status status = DAMACY_OK;

  CHECK_SILENT(Invalid, self);
  CHECK_SILENT(Invalid, samples);
  CHECK_SILENT(Invalid, out);
  CHECK_SILENT(Invalid, out->read_ops);
  CHECK_SILENT(Invalid, out->chunk_plans);
  CHECK_SILENT(Invalid, out->sample_plans);
  CHECK_SILENT(Invalid, out->paths);
  CHECK_SILENT(Invalid, dst_strides);
  CHECK_SILENT(Invalid, dst_full_rank >= 1);
  path_intern_reset(out->paths);
  out->n_read_ops = 0;
  out->n_chunk_plans = 0;
  out->n_sample_plans = 0;

  for (uint32_t sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
    const struct planner_sample* sample = &samples[sample_idx];
    if (!sample->uri) {
      status = DAMACY_INVAL;
      goto Cleanup;
    }

    const void* meta_value = NULL;
    int meta_err = 0;
    enum prefetch_state meta_state = prefetch_cache_query(
      self->cfg.array_meta_cache, sample->h_meta, &meta_value, &meta_err);
    if (meta_state == PREFETCH_STATE_PENDING) {
      // Batch gate should make this unreachable.
      log_error("planner: meta still PENDING (uri=%s)", sample->uri);
      status = DAMACY_INVAL;
      goto Cleanup;
    }
    if (meta_state == PREFETCH_STATE_ERROR) {
      status = meta_err ? (enum damacy_status)meta_err : DAMACY_INVAL;
      goto Cleanup;
    }
    const struct zarr_metadata* meta = (const struct zarr_metadata*)meta_value;
    if (!meta) {
      status = DAMACY_INVAL;
      goto Cleanup;
    }
    if (sample->aabb.rank != meta->rank) {
      status = DAMACY_RANK;
      goto Cleanup;
    }
    if ((uint8_t)(meta->rank + 1) != dst_full_rank) {
      status = DAMACY_RANK;
      goto Cleanup;
    }
    if (!cast_path_supported(self->cfg.dst_dtype, meta->dtype)) {
      status = DAMACY_DTYPE;
      goto Cleanup;
    }
    if (meta->inner_codec.id == CODEC_BLOSC_LZ4) {
      log_error("planner: blosc1-lz4 inner codec is not supported (uri=%s)",
                sample->uri);
      status = DAMACY_DECODE;
      goto Cleanup;
    }

    uint64_t inner_per_shard_dim[DAMACY_MAX_RANK];
    if (zarr_metadata_inner_per_shard(meta, inner_per_shard_dim, NULL)) {
      status = DAMACY_DECODE;
      goto Cleanup;
    }

    uint64_t decompressed_n_bytes = inner_chunk_bytes(meta);
    if (decompressed_n_bytes == 0) {
      status = DAMACY_DECODE;
      goto Cleanup;
    }
    if (self->cfg.max_chunk_uncompressed_bytes > 0 &&
        decompressed_n_bytes > self->cfg.max_chunk_uncompressed_bytes) {
      log_error("planner: chunk uncompressed=%llu exceeds runtime cap=%llu "
                "(uri=%s)",
                (unsigned long long)decompressed_n_bytes,
                (unsigned long long)self->cfg.max_chunk_uncompressed_bytes,
                sample->uri);
      status = DAMACY_BUDGET;
      goto Cleanup;
    }

    uint64_t chunk_beg[DAMACY_MAX_RANK];
    uint64_t chunk_end[DAMACY_MAX_RANK];
    if (chunk_range(&sample->aabb, meta, chunk_beg, chunk_end)) {
      status = DAMACY_INVAL;
      goto Cleanup;
    }

    if (out->n_sample_plans >= out->sample_plans_cap) {
      status = DAMACY_OOM;
      goto Cleanup;
    }
    struct sample_plan* sp = &out->sample_plans[out->n_sample_plans];
    *sp = (struct sample_plan){
      .batch_pool_slot = batch_pool_slot,
      .sample_idx_in_batch = (uint16_t)sample_idx,
      .rank = meta->rank,
      .src_dtype = (uint8_t)meta->dtype,
      .sample_dst_off_elems = (int64_t)sample_idx * dst_strides[0],
      .chunk_count = 0,
    };
    memcpy(sp->fill_value, meta->fill_value, sizeof sp->fill_value);
    int64_t src_strides[DAMACY_MAX_RANK];
    row_major_strides(meta->inner_chunk_shape, meta->rank, src_strides);
    uint32_t chunk_count = 1;
    for (uint8_t d = 0; d < meta->rank; ++d) {
      uint32_t S = (uint32_t)meta->inner_chunk_shape[d];
      uint32_t N = (uint32_t)(chunk_end[d] - chunk_beg[d]);
      int64_t chunk_grid_origin = (int64_t)(chunk_beg[d] * (uint64_t)S);
      sp->dims[d] = (struct sample_dim){
        .chunk_shape = S,
        .chunk_grid_extent = N,
        .aabb_lo_relative = sample->aabb.dims[d].beg - chunk_grid_origin,
        .aabb_extent = sample->aabb.dims[d].end - sample->aabb.dims[d].beg,
        .dst_stride = dst_strides[d + 1],
        .src_stride = src_strides[d],
      };
      chunk_count *= N;
    }
    sp->chunk_count = chunk_count;
    out->n_sample_plans++;

    const void* layout_value = NULL;
    int layout_err = 0;
    enum prefetch_state layout_state =
      prefetch_cache_query(self->cfg.chunk_layout_cache,
                           sample->h_layout,
                           &layout_value,
                           &layout_err);
    if (layout_state == PREFETCH_STATE_PENDING) {
      // Batch gate should make this unreachable.
      log_error("planner: chunk_layout still PENDING (uri=%s)", sample->uri);
      status = DAMACY_INVAL;
      goto Cleanup;
    }
    if (layout_state == PREFETCH_STATE_ERROR) {
      status = layout_err ? (enum damacy_status)layout_err : DAMACY_DECODE;
      goto Cleanup;
    }
    // NULL value on READY is legitimate for non-blosc codecs;
    // emit_chunk falls back to worst-case caps.
    const struct chunk_layout* layout =
      (const struct chunk_layout*)layout_value;

    struct emit_ctx ctx = {
      .sample = sample,
      .meta = meta,
      .inner_per_shard_dim = inner_per_shard_dim,
      .chunk_beg = chunk_beg,
      .sample_idx_in_batch = sample_idx,
      .batch_pool_slot = batch_pool_slot,
      .codec_id = (uint8_t)meta->inner_codec.id,
      .decompressed_n_bytes = (uint32_t)decompressed_n_bytes,
      .page_alignment_bytes = self->cfg.page_alignment,
      .sp = sp,
      .chunk_layout = layout,
      .max_substreams_per_chunk = self->cfg.max_substreams_per_chunk,
    };

    struct sample_shard_iterator shard_it;
    if (sample_shard_iterator_init(&shard_it, meta, &sample->aabb)) {
      status = DAMACY_INVAL;
      goto Cleanup;
    }

    uint64_t shard_coord[DAMACY_MAX_RANK];
    uint32_t shard_idx_in_sample = 0;
    while (sample_shard_iterator_next(&shard_it, shard_coord)) {
      if (shard_idx_in_sample >= sample->n_shards) {
        log_error(
          "planner: shard count mismatch (uri=%s iterated=%u expected=%u)",
          sample->uri,
          shard_idx_in_sample + 1u,
          sample->n_shards);
        status = DAMACY_INVAL;
        goto Cleanup;
      }
      struct prefetch_handle h = sample->h_shards[shard_idx_in_sample++];
      const struct shard_index_value* sv =
        (const struct shard_index_value*)prefetch_cache_try_get(
          self->cfg.shard_index_cache, h);
      if (!sv) {
        int err = 0;
        enum prefetch_state st =
          prefetch_cache_query(self->cfg.shard_index_cache, h, NULL, &err);
        if (st == PREFETCH_STATE_PENDING) {
          // Batch gate should have made this unreachable.
          log_error("planner: shard still PENDING (uri=%s)", sample->uri);
          status = DAMACY_INVAL;
          goto Cleanup;
        }
        if (err == DAMACY_NOTFOUND) {
          ctx.shard_missing = 1;
          ctx.shard_entries = NULL;
          ctx.n_shard_entries = 0;
          ctx.interned_path = NULL;
        } else {
          status = err ? (enum damacy_status)err : DAMACY_DECODE;
          goto Cleanup;
        }
      } else {
        ctx.shard_missing = 0;
        ctx.shard_entries = sv->entries;
        ctx.n_shard_entries = sv->n_entries;
        if (zarr_shard_path_build(
              &self->path_sb, sample->uri, shard_coord, meta->rank)) {
          status = DAMACY_OOM;
          goto Cleanup;
        }
        ctx.interned_path =
          path_intern_acquire(out->paths, strbuf_cstr(&self->path_sb));
        if (!ctx.interned_path) {
          status = DAMACY_OOM;
          goto Cleanup;
        }
      }

      uint64_t chunk_beg_in_shard[DAMACY_MAX_RANK];
      uint64_t chunk_end_in_shard[DAMACY_MAX_RANK];
      for (uint8_t d = 0; d < meta->rank; ++d) {
        uint64_t s_beg = shard_coord[d] * inner_per_shard_dim[d];
        uint64_t s_end = s_beg + inner_per_shard_dim[d];
        chunk_beg_in_shard[d] = s_beg > chunk_beg[d] ? s_beg : chunk_beg[d];
        chunk_end_in_shard[d] = s_end < chunk_end[d] ? s_end : chunk_end[d];
      }

      uint64_t chunk_coord[DAMACY_MAX_RANK];
      for (uint8_t d = 0; d < meta->rank; ++d)
        chunk_coord[d] = chunk_beg_in_shard[d];

      for (;;) {
        uint64_t local_inner[DAMACY_MAX_RANK];
        for (uint8_t d = 0; d < meta->rank; ++d)
          local_inner[d] = chunk_coord[d] % inner_per_shard_dim[d];

        enum damacy_status emit_status =
          emit_chunk(&ctx, chunk_coord, local_inner, out);
        if (emit_status != DAMACY_OK) {
          status = emit_status;
          goto Cleanup;
        }

        int finished = 1;
        for (int d = (int)meta->rank - 1; d >= 0; --d) {
          chunk_coord[d]++;
          if (chunk_coord[d] < chunk_end_in_shard[d]) {
            finished = 0;
            break;
          }
          chunk_coord[d] = chunk_beg_in_shard[d];
        }
        if (finished)
          break;
      }
    }
    if (shard_idx_in_sample != sample->n_shards) {
      log_error(
        "planner: shard count mismatch (uri=%s iterated=%u expected=%u)",
        sample->uri,
        shard_idx_in_sample,
        sample->n_shards);
      status = DAMACY_INVAL;
      goto Cleanup;
    }
  }

  {
    status = planner_ensure_scratch(self, out->n_read_ops);
    if (status != DAMACY_OK)
      goto Cleanup;
    status = coalesce_chunks(out,
                             self->cfg.read_op_max_bytes,
                             self->cfg.max_chunks_per_wave,
                             self->scratch_u32,
                             self->scratch_ops);
    if (status != DAMACY_OK)
      goto Cleanup;
    if (out->n_read_ops + 1u > 3u * self->scratch_u32_cap) {
      status = DAMACY_OOM;
      goto Cleanup;
    }
    status =
      group_chunks_by_read(out, self->scratch_u32, self->scratch_chunk_plans);
    if (status != DAMACY_OK)
      goto Cleanup;
  }

  return DAMACY_OK;

Cleanup:
  return status;

Invalid:
  return DAMACY_INVAL;
}
