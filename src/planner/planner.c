#include "planner/planner.h"

#include "dtype/dtype.h"
#include "log/log.h"
#include "planner/coalesce.h"
#include "planner/group_chunks.h"
#include "util/prelude.h"
#include "util/strbuf.h"
#include "zarr/zarr_meta_cache.h"
#include "zarr/zarr_metadata.h"
#include "zarr/zarr_shard_cache.h"
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

// Inner-chunk grid range for the chunks intersecting [aabb.lo, aabb.hi).
// chunk_lo/chunk_hi are in inner-chunk-grid units (half-open).
static int
chunk_range(const struct damacy_aabb* aabb,
            const struct zarr_metadata* meta,
            uint64_t* chunk_lo,
            uint64_t* chunk_hi)
{
  if (aabb->rank != meta->rank)
    return 1;
  for (uint8_t d = 0; d < meta->rank; ++d) {
    int64_t lo = aabb->dims[d].beg;
    int64_t hi = aabb->dims[d].end;
    if (lo < 0 || hi <= lo)
      return 1;
    if ((uint64_t)hi > meta->shape[d])
      return 1;
    uint64_t chunk_extent = meta->inner_chunk_shape[d];
    if (chunk_extent == 0)
      return 1;
    chunk_lo[d] = (uint64_t)lo / chunk_extent;
    chunk_hi[d] = ((uint64_t)hi - 1) / chunk_extent + 1;
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
  CHECK_SILENT(Error, cfg->meta_cache);
  CHECK_SILENT(Error, cfg->shard_cache);
  CHECK_SILENT(Error, cfg->page_alignment > 0);

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
  const struct damacy_sample* sample;
  const struct zarr_metadata* meta;
  const uint64_t* inner_per_shard_dim; // [meta->rank]
  const uint64_t* chunk_lo;            // [meta->rank], first chunk for sample
  uint32_t sample_idx_in_batch;
  uint16_t batch_pool_slot;
  uint8_t codec_id;
  uint32_t decompressed_n_bytes; // == inner_chunk_bytes(meta) (validated)
  uint64_t page_alignment_bytes;
  // Mutable handle to this sample's plan entry. emit_chunk lazily
  // populates ->layout / ->layout_probed on the first non-fill emit.
  struct sample_plan* sp;
  struct zarr_meta_cache* meta_cache;
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
  // Empty key is fine: peel skips IO submission for nbytes == 0 chunks.
  r->shard_path[0] = '\0';
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
    cp->chunk_d[d] = (uint32_t)(chunk_coord[d] - ctx->chunk_lo[d]);
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

  // First non-fill chunk for this sample: probe (or fetch cached) the
  // blosc1 chunk layout and stash it on the sample_plan. Probe failure
  // is non-fatal — downstream falls back to MAX_BLOCKS_PER_CHUNK in
  // cap calculations.
  if (ctx->sp && !ctx->sp->layout_probed) {
    struct chunk_layout cl = { 0 };
    if (zarr_meta_cache_probe_layout(ctx->meta_cache,
                                     ctx->sample->uri,
                                     ctx->interned_path,
                                     entry->offset,
                                     (uint32_t)entry->nbytes,
                                     ctx->codec_id,
                                     &cl) == 0) {
      ctx->sp->layout = cl;
      ctx->sp->layout_probed = 1;
    }
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
  size_t path_len = strlen(ctx->interned_path);
  if (path_len + 1 > sizeof r->shard_path)
    return DAMACY_OOM;
  memcpy(r->shard_path, ctx->interned_path, path_len + 1);
  r->file_offset = aligned_file_offset;
  r->nbytes = (uint32_t)read_n_bytes;
  r->host_buf_offset = 0;
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
    cp->chunk_d[d] = (uint32_t)(chunk_coord[d] - ctx->chunk_lo[d]);

  out->n_chunk_plans++;
  return DAMACY_OK;
}

enum damacy_status
planner_plan(struct planner* self,
             const struct damacy_sample* samples,
             uint32_t n_samples,
             uint16_t batch_pool_slot,
             const int64_t* dst_strides,
             uint8_t dst_full_rank,
             struct planner_output* out)
{
  // Pin held across shard-coord runs. Released on shard change and on
  // every exit path (Cleanup). Zero-init = unpinned, safe to release.
  struct zarr_shard_pin active_pin = { 0 };
  enum damacy_status status = DAMACY_OK;

  // Precondition checks before any pin could be held — these don't go
  // through Cleanup because `self` may be NULL (Cleanup dereferences
  // self->cfg.shard_cache).
  CHECK_SILENT(Invalid, self);
  CHECK_SILENT(Invalid, samples);
  CHECK_SILENT(Invalid, out);
  CHECK_SILENT(Invalid, out->read_ops);
  CHECK_SILENT(Invalid, out->chunk_plans);
  CHECK_SILENT(Invalid, out->sample_plans);
  CHECK_SILENT(Invalid, dst_strides);
  CHECK_SILENT(Invalid, dst_full_rank >= 1);
  out->n_read_ops = 0;
  out->n_chunk_plans = 0;
  out->n_sample_plans = 0;

  for (uint32_t sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
    const struct damacy_sample* sample = &samples[sample_idx];
    if (!sample->uri) {
      status = DAMACY_INVAL;
      goto Cleanup;
    }

    const struct zarr_metadata* meta = NULL;
    enum damacy_status meta_status =
      zarr_meta_cache_get(self->cfg.meta_cache, sample->uri, &meta);
    if (meta_status != DAMACY_OK) {
      status = meta_status;
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
    if (meta->inner_codec.id == CODEC_BLOSC_LZ4) {
      log_error("planner: blosc1-lz4 inner codec is not supported (uri=%s)",
                sample->uri);
      status = DAMACY_INVAL;
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
      status = DAMACY_INVAL;
      goto Cleanup;
    }

    uint64_t chunk_lo[DAMACY_MAX_RANK];
    uint64_t chunk_hi[DAMACY_MAX_RANK];
    if (chunk_range(&sample->aabb, meta, chunk_lo, chunk_hi)) {
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
      .chunk_count = 0, // filled after the chunk emit loop
    };
    memcpy(sp->fill_value, meta->fill_value, sizeof sp->fill_value);
    int64_t src_strides[DAMACY_MAX_RANK];
    row_major_strides(meta->inner_chunk_shape, meta->rank, src_strides);
    uint32_t chunk_count = 1;
    for (uint8_t d = 0; d < meta->rank; ++d) {
      uint32_t S = (uint32_t)meta->inner_chunk_shape[d];
      uint32_t N = (uint32_t)(chunk_hi[d] - chunk_lo[d]);
      int64_t chunk_grid_origin = (int64_t)(chunk_lo[d] * (uint64_t)S);
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

    struct emit_ctx ctx = {
      .sample = sample,
      .meta = meta,
      .inner_per_shard_dim = inner_per_shard_dim,
      .chunk_lo = chunk_lo,
      .sample_idx_in_batch = sample_idx,
      .batch_pool_slot = batch_pool_slot,
      .codec_id = (uint8_t)meta->inner_codec.id,
      .decompressed_n_bytes = (uint32_t)decompressed_n_bytes,
      .page_alignment_bytes = self->cfg.page_alignment,
      .sp = sp,
      .meta_cache = self->cfg.meta_cache,
    };

    // Iterate chunks in [chunk_lo, chunk_hi) row-major.
    uint64_t chunk_coord[DAMACY_MAX_RANK];
    for (uint8_t d = 0; d < meta->rank; ++d)
      chunk_coord[d] = chunk_lo[d];

    // Track last-seen shard so we skip the cache+path work for chunks
    // in the same shard.
    uint64_t cached_shard_coord[DAMACY_MAX_RANK] = { 0 };
    int have_cached_shard = 0;

    for (;;) {
      // Compute shard coord and local-inner-coord for this chunk.
      uint64_t shard_coord[DAMACY_MAX_RANK];
      uint64_t local_inner[DAMACY_MAX_RANK];
      for (uint8_t d = 0; d < meta->rank; ++d) {
        shard_coord[d] = chunk_coord[d] / inner_per_shard_dim[d];
        local_inner[d] = chunk_coord[d] % inner_per_shard_dim[d];
      }

      // Same shard as cached?
      int same_shard = have_cached_shard;
      for (uint8_t d = 0; d < meta->rank && same_shard; ++d)
        if (shard_coord[d] != cached_shard_coord[d])
          same_shard = 0;

      if (!same_shard) {
        // Crossing into a new shard: drop the prior pin (if any)
        // before acquiring the next one. release is NULL-safe.
        zarr_shard_cache_release(self->cfg.shard_cache, active_pin);
        active_pin = (struct zarr_shard_pin){ 0 };

        enum damacy_status shard_status =
          zarr_shard_cache_get(self->cfg.shard_cache,
                               sample->uri,
                               meta,
                               shard_coord,
                               &active_pin,
                               &ctx.shard_entries,
                               &ctx.n_shard_entries);
        if (shard_status == DAMACY_NOTFOUND) {
          // Shard file absent: emit fill chunks for the whole shard.
          ctx.shard_missing = 1;
          ctx.shard_entries = NULL;
          ctx.n_shard_entries = 0;
          ctx.interned_path = NULL;
        } else if (shard_status != DAMACY_OK) {
          status = shard_status;
          goto Cleanup;
        } else {
          ctx.shard_missing = 0;
          if (zarr_shard_path_build(
                &self->path_sb, sample->uri, shard_coord, meta->rank)) {
            status = DAMACY_OOM;
            goto Cleanup;
          }
          // emit_chunk copies path bytes into each read_op; path_sb is
          // overwritten next iteration but each read_op carries its own.
          ctx.interned_path = strbuf_cstr(&self->path_sb);
        }
        for (uint8_t d = 0; d < meta->rank; ++d)
          cached_shard_coord[d] = shard_coord[d];
        have_cached_shard = 1;
      }

      enum damacy_status emit_status =
        emit_chunk(&ctx, chunk_coord, local_inner, out);
      if (emit_status != DAMACY_OK) {
        status = emit_status;
        goto Cleanup;
      }

      // Advance row-major.
      int finished = 1;
      for (int d = (int)meta->rank - 1; d >= 0; --d) {
        chunk_coord[d]++;
        if (chunk_coord[d] < chunk_hi[d]) {
          finished = 0;
          break;
        }
        chunk_coord[d] = chunk_lo[d];
      }
      if (finished)
        break;
    }
  }

  // Release the active pin before post-loop planning. Coalesce /
  // group_chunks operate on read_ops/chunk_plans only — shard_entries
  // are no longer referenced.
  zarr_shard_cache_release(self->cfg.shard_cache, active_pin);
  active_pin = (struct zarr_shard_pin){ 0 };

  {
    status = planner_ensure_scratch(self, out->n_read_ops);
    if (status != DAMACY_OK)
      goto Cleanup;
    status = coalesce_chunks(
      out, self->cfg.read_op_max_bytes, self->scratch_u32, self->scratch_ops);
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
  zarr_shard_cache_release(self->cfg.shard_cache, active_pin);
  return status;

Invalid:
  return DAMACY_INVAL;
}
