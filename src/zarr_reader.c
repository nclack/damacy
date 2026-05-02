#include "zarr.h"

#include "dimension.h"
#include "limits.h"
#include "store.h"
#include "util/strbuf.h"
#include "zarr_metadata.h"
#include "zarr_shard_index.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

struct shard_index_slot
{
  uint64_t shard_linear;            // row-major shard coordinate
  char* key;                        // owned; "c/<a>/<b>/..."
  struct zarr_shard_entry* entries; // owned; n_inner_per_shard entries
  size_t max_compressed;            // largest entry in this shard
};

struct zarr_reader
{
  struct store* store; // borrowed
  char* prefix;        // owned

  struct zarr_metadata meta;
  struct dimension dims[DAMACY_MAX_RANK];
  struct zarr_array_info info;

  // Inner chunks per shard along each dim, and total per shard.
  uint64_t inner_per_shard_dim[DAMACY_MAX_RANK];
  uint64_t inner_per_shard_total;

  // Shard grid extents (n shards along each dim) and total shard count.
  uint64_t shard_grid[DAMACY_MAX_RANK];
  uint64_t n_shards_total;

  // Shard index cache. Linear scan; small until profiled.
  pthread_mutex_t cache_mu;
  struct shard_index_slot* slots;
  size_t n_slots;
  size_t cap_slots;
  size_t max_compressed_seen;

  size_t inner_chunk_uncompressed_bytes;
};

static char*
str_dup_n(const char* s)
{
  if (!s)
    return NULL;
  size_t n = strlen(s);
  char* p = (char*)malloc(n + 1);
  if (!p)
    return NULL;
  memcpy(p, s, n + 1);
  return p;
}

// Build "<prefix>/<file>" or just "<file>" when prefix is empty.
static int
join_meta_key(struct strbuf* sb, const char* prefix, const char* file)
{
  strbuf_reset(sb);
  if (prefix && prefix[0]) {
    if (strbuf_append_cstr(sb, prefix))
      return 1;
    size_t L = strbuf_len(sb);
    if (L > 0 && strbuf_cstr(sb)[L - 1] != '/') {
      if (strbuf_append(sb, "/", 1))
        return 1;
    }
  }
  return strbuf_append_cstr(sb, file);
}

// Build "<prefix>/c/<i0>/<i1>/.../<ik-1>".
static int
build_shard_key(struct strbuf* sb,
                const char* prefix,
                const uint64_t* shard_coord,
                uint8_t rank)
{
  strbuf_reset(sb);
  if (prefix && prefix[0]) {
    if (strbuf_append_cstr(sb, prefix))
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

// Inner-chunk linear index within a shard from per-dim inner offsets
// (row-major over inner_per_shard_dim).
static uint64_t
inner_linear(const uint64_t* per_shard_dim,
             const uint64_t* inner_off,
             uint8_t rank)
{
  uint64_t lin = 0;
  for (uint8_t d = 0; d < rank; ++d)
    lin = lin * per_shard_dim[d] + inner_off[d];
  return lin;
}

// Look up or load the shard index for a given linear shard coord.
// Returns NULL on failure.
static struct shard_index_slot*
get_shard_slot(struct zarr_reader* r,
               uint64_t shard_linear,
               const uint64_t* shard_coord)
{
  pthread_mutex_lock(&r->cache_mu);
  for (size_t i = 0; i < r->n_slots; ++i) {
    if (r->slots[i].shard_linear == shard_linear) {
      struct shard_index_slot* slot = &r->slots[i];
      pthread_mutex_unlock(&r->cache_mu);
      return slot;
    }
  }
  pthread_mutex_unlock(&r->cache_mu);

  // Build key + load index footer.
  struct strbuf key = { 0 };
  if (build_shard_key(&key, r->prefix, shard_coord, r->meta.rank)) {
    strbuf_free(&key);
    return NULL;
  }

  uint64_t file_size = 0;
  if (store_stat(r->store, strbuf_cstr(&key), &file_size) || file_size == 0) {
    strbuf_free(&key);
    return NULL;
  }

  size_t idx_bytes = zarr_shard_index_size(r->inner_per_shard_total);
  if (file_size < (uint64_t)idx_bytes) {
    strbuf_free(&key);
    return NULL;
  }
  uint64_t idx_off =
    r->meta.index_location_end ? (file_size - (uint64_t)idx_bytes) : 0;

  void* footer = malloc(idx_bytes);
  if (!footer) {
    strbuf_free(&key);
    return NULL;
  }
  struct store_read sr = {
    .key = strbuf_cstr(&key),
    .dst = footer,
    .offset = idx_off,
    .len = idx_bytes,
  };
  if (store_read_many(r->store, &sr, 1)) {
    free(footer);
    strbuf_free(&key);
    return NULL;
  }

  struct zarr_shard_entry* entries = (struct zarr_shard_entry*)calloc(
    r->inner_per_shard_total, sizeof(struct zarr_shard_entry));
  if (!entries) {
    free(footer);
    strbuf_free(&key);
    return NULL;
  }
  if (zarr_shard_index_parse(
        footer, idx_bytes, r->inner_per_shard_total, entries)) {
    free(footer);
    free(entries);
    strbuf_free(&key);
    return NULL;
  }
  free(footer);

  size_t local_max = 0;
  for (uint64_t i = 0; i < r->inner_per_shard_total; ++i) {
    if (entries[i].nbytes != ZARR_SHARD_EMPTY_NBYTES &&
        entries[i].nbytes > local_max)
      local_max = (size_t)entries[i].nbytes;
  }

  // Insert into cache (race: another caller may have inserted; in that case
  // discard our copy and reuse theirs).
  pthread_mutex_lock(&r->cache_mu);
  for (size_t i = 0; i < r->n_slots; ++i) {
    if (r->slots[i].shard_linear == shard_linear) {
      struct shard_index_slot* slot = &r->slots[i];
      pthread_mutex_unlock(&r->cache_mu);
      free(entries);
      strbuf_free(&key);
      return slot;
    }
  }
  if (r->n_slots == r->cap_slots) {
    size_t new_cap = r->cap_slots ? r->cap_slots * 2 : 16;
    struct shard_index_slot* p = (struct shard_index_slot*)realloc(
      r->slots, new_cap * sizeof(struct shard_index_slot));
    if (!p) {
      pthread_mutex_unlock(&r->cache_mu);
      free(entries);
      strbuf_free(&key);
      return NULL;
    }
    r->slots = p;
    r->cap_slots = new_cap;
  }
  char* key_owned = str_dup_n(strbuf_cstr(&key));
  if (!key_owned) {
    pthread_mutex_unlock(&r->cache_mu);
    free(entries);
    strbuf_free(&key);
    return NULL;
  }
  strbuf_free(&key);
  r->slots[r->n_slots].shard_linear = shard_linear;
  r->slots[r->n_slots].key = key_owned;
  r->slots[r->n_slots].entries = entries;
  r->slots[r->n_slots].max_compressed = local_max;
  if (local_max > r->max_compressed_seen)
    r->max_compressed_seen = local_max;
  struct shard_index_slot* out = &r->slots[r->n_slots];
  r->n_slots++;
  pthread_mutex_unlock(&r->cache_mu);
  return out;
}

struct zarr_reader*
zarr_reader_open(const struct zarr_reader_config* cfg)
{
  if (!cfg || !cfg->store || !cfg->prefix)
    return NULL;

  struct zarr_reader* r = (struct zarr_reader*)calloc(1, sizeof(*r));
  if (!r)
    return NULL;
  r->store = cfg->store;
  r->prefix = str_dup_n(cfg->prefix);
  if (!r->prefix)
    goto fail;
  pthread_mutex_init(&r->cache_mu, NULL);

  struct strbuf key = { 0 };
  if (join_meta_key(&key, r->prefix, "zarr.json")) {
    strbuf_free(&key);
    goto fail;
  }
  struct store_view view = { 0 };
  int rc = store_map(cfg->store, strbuf_cstr(&key), &view);
  strbuf_free(&key);
  if (rc)
    goto fail;

  rc = zarr_metadata_parse((const char*)view.data, view.len, &r->meta);
  store_unmap(cfg->store, &view);
  if (rc)
    goto fail;
  if (!r->meta.sharded || r->meta.inner_codec.id != CODEC_ZSTD)
    goto fail;

  // Populate dims and the public info.
  r->inner_per_shard_total = 1;
  r->n_shards_total = 1;
  for (uint8_t d = 0; d < r->meta.rank; ++d) {
    r->dims[d].size = r->meta.shape[d];
    r->dims[d].chunk_size = r->meta.inner_chunk_shape[d];
    r->dims[d].chunks_per_shard =
      r->meta.shard_shape[d] / r->meta.inner_chunk_shape[d];
    r->dims[d].name = NULL;
    r->dims[d].downsample = 0;
    r->dims[d].storage_position = d;

    r->inner_per_shard_dim[d] = r->dims[d].chunks_per_shard;
    r->inner_per_shard_total *= r->inner_per_shard_dim[d];

    // Number of shards along this dim = ceil(shape / shard_shape).
    uint64_t shard_extent = r->meta.shard_shape[d];
    r->shard_grid[d] = (r->meta.shape[d] + shard_extent - 1) / shard_extent;
    r->n_shards_total *= r->shard_grid[d];
  }
  r->info.dtype = r->meta.dtype;
  r->info.rank = r->meta.rank;
  r->info.dims = r->dims;
  r->info.codec = r->meta.inner_codec;

  size_t voxels = 1;
  for (uint8_t d = 0; d < r->meta.rank; ++d)
    voxels *= (size_t)r->meta.inner_chunk_shape[d];
  r->inner_chunk_uncompressed_bytes = voxels * dtype_bpe(r->meta.dtype);

  return r;

fail:
  if (r) {
    pthread_mutex_destroy(&r->cache_mu);
    free(r->prefix);
    free(r);
  }
  return NULL;
}

void
zarr_reader_close(struct zarr_reader* r)
{
  if (!r)
    return;
  for (size_t i = 0; i < r->n_slots; ++i) {
    free(r->slots[i].key);
    free(r->slots[i].entries);
  }
  free(r->slots);
  pthread_mutex_destroy(&r->cache_mu);
  free(r->prefix);
  free(r);
}

const struct zarr_array_info*
zarr_reader_info(const struct zarr_reader* r)
{
  return r ? &r->info : NULL;
}

int
zarr_reader_locate(struct zarr_reader* r,
                   const int64_t* chunk_coord,
                   struct zarr_chunk_loc* out)
{
  if (!r || !chunk_coord || !out)
    return 1;

  uint64_t shard_coord[DAMACY_MAX_RANK];
  uint64_t inner_off[DAMACY_MAX_RANK];
  uint64_t shard_linear = 0;

  for (uint8_t d = 0; d < r->meta.rank; ++d) {
    if (chunk_coord[d] < 0)
      return 1;
    uint64_t c = (uint64_t)chunk_coord[d];
    uint64_t cps = r->dims[d].chunks_per_shard;
    if (cps == 0)
      return 1;
    shard_coord[d] = c / cps;
    inner_off[d] = c % cps;
    if (shard_coord[d] >= r->shard_grid[d])
      return 1;
    shard_linear = shard_linear * r->shard_grid[d] + shard_coord[d];
  }

  struct shard_index_slot* slot = get_shard_slot(r, shard_linear, shard_coord);
  if (!slot)
    return 1;
  uint64_t lin = inner_linear(r->inner_per_shard_dim, inner_off, r->meta.rank);
  if (lin >= r->inner_per_shard_total)
    return 1;
  struct zarr_shard_entry e = slot->entries[lin];
  if (e.offset == ZARR_SHARD_EMPTY_OFFSET ||
      e.nbytes == ZARR_SHARD_EMPTY_NBYTES)
    return 1;
  out->key = slot->key;
  out->offset = e.offset;
  out->len = (size_t)e.nbytes;
  return 0;
}

size_t
zarr_reader_chunk_max_compressed_bytes(const struct zarr_reader* r)
{
  if (!r)
    return 0;
  return r->max_compressed_seen;
}

size_t
zarr_reader_chunk_uncompressed_bytes(const struct zarr_reader* r)
{
  return r ? r->inner_chunk_uncompressed_bytes : 0;
}
