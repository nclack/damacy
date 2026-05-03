#include "planner.h"

#include "dtype.h"
#include "util/prelude.h"
#include "util/strbuf.h"
#include "zarr_meta_cache.h"
#include "zarr_metadata.h"
#include "zarr_shard_cache.h"
#include "zarr_shard_index.h"

#include <stdlib.h>
#include <string.h>

struct planner
{
  struct planner_config cfg;
  // Path strings owned by this planner; freed at the start of each
  // planner_plan call (and on destroy).
  char** owned_paths;
  uint32_t n_owned_paths;
  uint32_t cap_owned_paths;
  struct strbuf path_sb;
};

// --- ownership helpers ---------------------------------------------------

static void
clear_owned_paths(struct planner* p)
{
  for (uint32_t i = 0; i < p->n_owned_paths; ++i)
    free(p->owned_paths[i]);
  p->n_owned_paths = 0;
}

// Take ownership of a NUL-terminated copy of s. Returns NULL on alloc
// failure. Stored pointer is stable until the next clear_owned_paths().
static const char*
take_path(struct planner* p, const char* s)
{
  if (p->n_owned_paths == p->cap_owned_paths) {
    uint32_t new_cap = p->cap_owned_paths ? p->cap_owned_paths * 2 : 16;
    char** nb = (char**)realloc(p->owned_paths, new_cap * sizeof(char*));
    if (!nb)
      return NULL;
    p->owned_paths = nb;
    p->cap_owned_paths = new_cap;
  }
  size_t n = strlen(s);
  char* copy = (char*)malloc(n + 1);
  if (!copy)
    return NULL;
  memcpy(copy, s, n + 1);
  p->owned_paths[p->n_owned_paths++] = copy;
  return copy;
}

// Build "<uri>/c/<a>/<b>/...".
static int
build_shard_path(struct strbuf* sb,
                 const char* uri,
                 const uint64_t* shard_coord,
                 uint8_t rank)
{
  strbuf_reset(sb);
  if (uri && uri[0]) {
    if (strbuf_append_cstr(sb, uri))
      return 1;
    size_t L = strbuf_len(sb);
    if (L > 0 && strbuf_cstr(sb)[L - 1] != '/') {
      if (strbuf_append(sb, "/", 1))
        return 1;
    }
  }
  if (strbuf_append_cstr(sb, "c"))
    return 1;
  for (uint8_t d = 0; d < rank; ++d) {
    if (strbuf_appendf(sb, "/%llu", (unsigned long long)shard_coord[d]))
      return 1;
  }
  return 0;
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
    uint64_t cs = meta->inner_chunk_shape[d];
    if (cs == 0)
      return 1;
    chunk_lo[d] = (uint64_t)lo / cs;
    chunk_hi[d] = ((uint64_t)hi - 1) / cs + 1;
  }
  return 0;
}

// Row-major linear index into a multi-d grid given per-dim extents.
static uint64_t
row_major_linear(const uint64_t* coord, const uint64_t* extents, uint8_t rank)
{
  uint64_t lin = 0;
  for (uint8_t d = 0; d < rank; ++d)
    lin = lin * extents[d] + coord[d];
  return lin;
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

// Bytes per inner chunk (uncompressed).
static uint64_t
inner_chunk_bytes(const struct zarr_metadata* meta)
{
  uint64_t v = dtype_bpe(meta->dtype);
  for (uint8_t d = 0; d < meta->rank; ++d)
    v *= meta->inner_chunk_shape[d];
  return v;
}

static uint64_t
align_down_u64(uint64_t v, uint64_t a)
{
  return (v / a) * a;
}

static uint64_t
align_up_u64(uint64_t v, uint64_t a)
{
  return ((v + a - 1) / a) * a;
}

// --- public API ----------------------------------------------------------

enum damacy_status
planner_create(const struct planner_config* cfg, struct planner** out)
{
  struct planner* p = NULL;
  enum damacy_status status = DAMACY_INVAL;

  CHECK_SILENT(error, out);
  *out = NULL;
  CHECK_SILENT(error, cfg);
  CHECK_SILENT(error, cfg->meta_cache);
  CHECK_SILENT(error, cfg->shard_cache);
  CHECK_SILENT(error, cfg->page_alignment > 0);

  status = DAMACY_OOM;
  p = (struct planner*)calloc(1, sizeof(*p));
  CHECK(error, p);

  // Designated init zeroes any fields not explicitly named, including
  // path_sb (zero-init is documented as safe for struct strbuf).
  *p = (struct planner){ .cfg = *cfg };
  *out = p;
  return DAMACY_OK;

error:
  free(p);
  return status;
}

void
planner_destroy(struct planner* p)
{
  if (!p)
    return;
  clear_owned_paths(p);
  free(p->owned_paths);
  strbuf_free(&p->path_sb);
  free(p);
}

// Process one chunk: build read_op + chunk_plan, append to output.
// Returns DAMACY_OOM if the output buffers are full.
static enum damacy_status
emit_chunk(struct planner* p,
           const struct damacy_sample* sample,
           uint32_t sample_idx_in_batch,
           uint16_t batch_pool_slot,
           const struct zarr_metadata* meta,
           const uint64_t* chunk_coord,
           const uint64_t* local_inner,
           const uint64_t* inner_per_shard_dim,
           const struct zarr_shard_entry* shard_entries,
           uint64_t n_shard_entries,
           const char* interned_path,
           struct planner_output* out)
{
  uint64_t lin = row_major_linear(local_inner, inner_per_shard_dim, meta->rank);
  if (lin >= n_shard_entries)
    return DAMACY_DECODE;

  const struct zarr_shard_entry* e = &shard_entries[lin];
  if (e->offset == ZARR_SHARD_EMPTY_OFFSET ||
      e->nbytes == ZARR_SHARD_EMPTY_NBYTES)
    return DAMACY_OK; // empty chunk, skip

  if (e->nbytes > DAMACY_MAX_CHUNK_BYTES)
    return DAMACY_DECODE;

  // Page-aligned read window enclosing [offset, offset + nbytes).
  uint64_t pa = p->cfg.page_alignment;
  uint64_t file_off_aligned = align_down_u64(e->offset, pa);
  uint64_t end_aligned = align_up_u64(e->offset + e->nbytes, pa);
  uint64_t read_nbytes = end_aligned - file_off_aligned;
  if (read_nbytes > DAMACY_MAX_CHUNK_BYTES)
    return DAMACY_DECODE;
  uint32_t chunk_offset_in_read = (uint32_t)(e->offset - file_off_aligned);

  // Intersection of sample AABB with this chunk's level-0 footprint.
  // chunk_origin_l0[d] = chunk_coord[d] * inner_chunk_shape[d]
  int64_t inter_lo[DAMACY_MAX_RANK];
  int64_t inter_hi[DAMACY_MAX_RANK];
  for (uint8_t d = 0; d < meta->rank; ++d) {
    int64_t corigin = (int64_t)(chunk_coord[d] * meta->inner_chunk_shape[d]);
    int64_t cend = corigin + (int64_t)meta->inner_chunk_shape[d];
    int64_t alo = sample->aabb.dims[d].beg;
    int64_t ahi = sample->aabb.dims[d].end;
    inter_lo[d] = alo > corigin ? alo : corigin;
    inter_hi[d] = ahi < cend ? ahi : cend;
    if (inter_lo[d] >= inter_hi[d])
      return DAMACY_OK; // shouldn't happen given chunk_range, but defensive
  }

  if (out->n_read_ops >= out->read_ops_cap ||
      out->n_chunk_plans >= out->chunk_plans_cap)
    return DAMACY_OOM;

  uint32_t read_idx = out->n_read_ops;
  out->read_ops[read_idx] = (struct read_op){
    .shard_path = interned_path,
    .file_offset = file_off_aligned,
    .nbytes = (uint32_t)read_nbytes,
    ._pad0 = 0,
    .dst_buf_offset = 0,
  };
  out->n_read_ops++;

  struct chunk_plan* cp = &out->chunk_plans[out->n_chunk_plans];
  *cp = (struct chunk_plan){
    .read_op_idx = read_idx,
    .offset_in_read = chunk_offset_in_read,
    .compressed_nbytes = (uint32_t)e->nbytes,
    .decompressed_nbytes = (uint32_t)inner_chunk_bytes(meta),
    .dev_decompressed_offset = 0,
    .batch_pool_slot = batch_pool_slot,
    ._pad0 = 0,
    ._pad1 = 0,
    .src = { .rank = meta->rank },
    .dst = { .rank = (uint8_t)(meta->rank + 1) },
    .src_strides = { 0 },
  };

  // src AABB: intersection in chunk-local coordinates.
  for (uint8_t d = 0; d < meta->rank; ++d) {
    int64_t corigin = (int64_t)(chunk_coord[d] * meta->inner_chunk_shape[d]);
    cp->src.dims[d] = (struct damacy_interval){
      .beg = inter_lo[d] - corigin,
      .end = inter_hi[d] - corigin,
    };
  }

  // dst AABB: leading sample-index axis, then intersection in
  // sample-local coordinates.
  cp->dst.dims[0] = (struct damacy_interval){
    .beg = sample_idx_in_batch,
    .end = sample_idx_in_batch + 1,
  };
  for (uint8_t d = 0; d < meta->rank; ++d) {
    cp->dst.dims[d + 1] = (struct damacy_interval){
      .beg = inter_lo[d] - sample->aabb.dims[d].beg,
      .end = inter_hi[d] - sample->aabb.dims[d].beg,
    };
  }

  // Element strides for the chunk (row-major over inner_chunk_shape).
  row_major_strides(meta->inner_chunk_shape, meta->rank, cp->src_strides);

  out->n_chunk_plans++;
  return DAMACY_OK;
}

enum damacy_status
planner_plan(struct planner* p,
             const struct damacy_sample* samples,
             uint32_t n_samples,
             uint16_t batch_pool_slot,
             struct planner_output* out)
{
  if (!p || !samples || !out)
    return DAMACY_INVAL;
  if (!out->read_ops || !out->chunk_plans)
    return DAMACY_INVAL;
  out->n_read_ops = 0;
  out->n_chunk_plans = 0;
  clear_owned_paths(p);

  for (uint32_t si = 0; si < n_samples; ++si) {
    const struct damacy_sample* s = &samples[si];
    if (!s->uri)
      return DAMACY_INVAL;

    const struct zarr_metadata* meta = NULL;
    enum damacy_status ms =
      zarr_meta_cache_get(p->cfg.meta_cache, s->uri, &meta);
    if (ms != DAMACY_OK)
      return ms;
    if (s->aabb.rank != meta->rank)
      return DAMACY_RANK;

    uint64_t inner_per_shard_dim[DAMACY_MAX_RANK];
    for (uint8_t d = 0; d < meta->rank; ++d) {
      if (meta->inner_chunk_shape[d] == 0 ||
          meta->shard_shape[d] % meta->inner_chunk_shape[d] != 0)
        return DAMACY_DECODE;
      inner_per_shard_dim[d] =
        meta->shard_shape[d] / meta->inner_chunk_shape[d];
    }

    uint64_t chunk_lo[DAMACY_MAX_RANK];
    uint64_t chunk_hi[DAMACY_MAX_RANK];
    if (chunk_range(&s->aabb, meta, chunk_lo, chunk_hi))
      return DAMACY_INVAL;

    // Iterate chunks in [chunk_lo, chunk_hi) row-major.
    uint64_t chunk[DAMACY_MAX_RANK];
    for (uint8_t d = 0; d < meta->rank; ++d)
      chunk[d] = chunk_lo[d];

    // Cache last shard we looked at so we don't re-fetch within a sample
    // for chunks in the same shard.
    uint64_t cached_shard[DAMACY_MAX_RANK] = { 0 };
    int have_cached = 0;
    const struct zarr_shard_entry* cached_entries = NULL;
    uint64_t cached_n_entries = 0;
    const char* cached_path = NULL;

    for (;;) {
      // Compute shard coord and local-inner-coord for this chunk.
      uint64_t shard_coord[DAMACY_MAX_RANK];
      uint64_t local_inner[DAMACY_MAX_RANK];
      for (uint8_t d = 0; d < meta->rank; ++d) {
        shard_coord[d] = chunk[d] / inner_per_shard_dim[d];
        local_inner[d] = chunk[d] % inner_per_shard_dim[d];
      }

      // Same shard as cached?
      int same = have_cached;
      for (uint8_t d = 0; d < meta->rank && same; ++d)
        if (shard_coord[d] != cached_shard[d])
          same = 0;

      if (!same) {
        enum damacy_status ss = zarr_shard_cache_get(p->cfg.shard_cache,
                                                     s->uri,
                                                     meta,
                                                     shard_coord,
                                                     &cached_entries,
                                                     &cached_n_entries);
        if (ss != DAMACY_OK)
          return ss;
        if (build_shard_path(&p->path_sb, s->uri, shard_coord, meta->rank))
          return DAMACY_OOM;
        cached_path = take_path(p, strbuf_cstr(&p->path_sb));
        if (!cached_path)
          return DAMACY_OOM;
        for (uint8_t d = 0; d < meta->rank; ++d)
          cached_shard[d] = shard_coord[d];
        have_cached = 1;
      }

      enum damacy_status es = emit_chunk(p,
                                         s,
                                         si,
                                         batch_pool_slot,
                                         meta,
                                         chunk,
                                         local_inner,
                                         inner_per_shard_dim,
                                         cached_entries,
                                         cached_n_entries,
                                         cached_path,
                                         out);
      if (es != DAMACY_OK)
        return es;

      // Advance row-major.
      int finished = 1;
      for (int d = (int)meta->rank - 1; d >= 0; --d) {
        chunk[d]++;
        if (chunk[d] < chunk_hi[d]) {
          finished = 0;
          break;
        }
        chunk[d] = chunk_lo[d];
      }
      if (finished)
        break;
    }
  }

  return DAMACY_OK;
}
