#include "zarr_shard_cache.h"

#include "limits.h"
#include "store.h"
#include "util/hash.h"
#include "util/lru.h"
#include "util/strbuf.h"
#include "zarr_metadata.h"
#include "zarr_shard_index.h"

#include <stdlib.h>
#include <string.h>

struct shard_entry
{
  char* uri; // owned, NUL-terminated
  uint8_t rank;
  uint64_t shard_coord[DAMACY_MAX_RANK];
  uint64_t n_entries;
  struct zarr_shard_entry* entries; // owned; n_entries elements
};

struct shard_probe
{
  const char* uri;
  uint8_t rank;
  const uint64_t* shard_coord;
};

struct zarr_shard_cache
{
  struct store* store; // borrowed
  struct lru* lru;
};

static char*
str_dup(const char* s)
{
  size_t n = strlen(s);
  char* p = (char*)malloc(n + 1);
  if (!p)
    return NULL;
  memcpy(p, s, n + 1);
  return p;
}

static uint64_t
shard_hash(const char* uri, const uint64_t* shard_coord, uint8_t rank)
{
  uint64_t h = hash_fnv1a_str(uri);
  return hash_combine(h,
                      hash_fnv1a(shard_coord, (size_t)rank * sizeof(uint64_t)));
}

static int
shard_eq(const void* value, const void* probe_key, void* user)
{
  (void)user;
  const struct shard_entry* e = (const struct shard_entry*)value;
  const struct shard_probe* p = (const struct shard_probe*)probe_key;
  if (e->rank != p->rank)
    return 0;
  if (strcmp(e->uri, p->uri) != 0)
    return 0;
  for (uint8_t d = 0; d < e->rank; ++d)
    if (e->shard_coord[d] != p->shard_coord[d])
      return 0;
  return 1;
}

static void
shard_destroy(void* value, void* user)
{
  (void)user;
  struct shard_entry* e = (struct shard_entry*)value;
  if (!e)
    return;
  free(e->uri);
  free(e->entries);
  free(e);
}

// Build "<uri>/c/<a>/<b>/...".
static int
build_shard_key(struct strbuf* sb,
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

// Inner-chunks-per-shard along each dim and the total. Returns 0 on
// success; non-zero if the shard shape isn't a clean multiple of the
// inner shape.
static int
inner_per_shard_total(const struct zarr_metadata* meta, uint64_t* out_total)
{
  uint64_t total = 1;
  for (uint8_t d = 0; d < meta->rank; ++d) {
    if (meta->inner_chunk_shape[d] == 0)
      return 1;
    if (meta->shard_shape[d] % meta->inner_chunk_shape[d] != 0)
      return 1;
    total *= meta->shard_shape[d] / meta->inner_chunk_shape[d];
  }
  *out_total = total;
  return 0;
}

struct zarr_shard_cache*
zarr_shard_cache_create(struct store* store, uint32_t capacity)
{
  if (!store || capacity == 0)
    return NULL;
  struct zarr_shard_cache* c = (struct zarr_shard_cache*)calloc(1, sizeof(*c));
  if (!c)
    return NULL;
  c->store = store;
  struct lru_ops ops = {
    .eq = shard_eq,
    .destroy = shard_destroy,
    .user = NULL,
  };
  c->lru = lru_create(capacity, 16, &ops);
  if (!c->lru) {
    free(c);
    return NULL;
  }
  return c;
}

void
zarr_shard_cache_destroy(struct zarr_shard_cache* c)
{
  if (!c)
    return;
  lru_destroy(c->lru);
  free(c);
}

enum damacy_status
zarr_shard_cache_get(struct zarr_shard_cache* c,
                     const char* uri,
                     const struct zarr_metadata* meta,
                     const uint64_t* shard_coord,
                     const struct zarr_shard_entry** out_entries,
                     uint64_t* out_n_entries)
{
  if (!c || !uri || !meta || !shard_coord || !out_entries || !out_n_entries)
    return DAMACY_INVAL;
  if (meta->rank == 0 || meta->rank > DAMACY_MAX_RANK)
    return DAMACY_RANK;
  *out_entries = NULL;
  *out_n_entries = 0;

  struct shard_probe probe = {
    .uri = uri,
    .rank = meta->rank,
    .shard_coord = shard_coord,
  };
  uint64_t h = shard_hash(uri, shard_coord, meta->rank);

  struct lru_entry* le = lru_get(c->lru, h, &probe);
  if (le) {
    const struct shard_entry* se =
      (const struct shard_entry*)lru_entry_value(le);
    *out_entries = se->entries;
    *out_n_entries = se->n_entries;
    return DAMACY_OK;
  }

  uint64_t n_inner = 0;
  if (inner_per_shard_total(meta, &n_inner))
    return DAMACY_INVAL;

  struct strbuf key = { 0 };
  if (build_shard_key(&key, uri, shard_coord, meta->rank)) {
    strbuf_free(&key);
    return DAMACY_OOM;
  }

  uint64_t file_size = 0;
  if (store_stat(c->store, strbuf_cstr(&key), &file_size) || file_size == 0) {
    strbuf_free(&key);
    return DAMACY_NOTFOUND;
  }

  size_t idx_bytes = zarr_shard_index_size((size_t)n_inner);
  if (file_size < (uint64_t)idx_bytes) {
    strbuf_free(&key);
    return DAMACY_DECODE;
  }
  uint64_t idx_off =
    meta->index_location_end ? (file_size - (uint64_t)idx_bytes) : 0;

  void* footer = malloc(idx_bytes);
  if (!footer) {
    strbuf_free(&key);
    return DAMACY_OOM;
  }
  struct store_read sr = {
    .key = strbuf_cstr(&key),
    .dst = footer,
    .offset = idx_off,
    .len = idx_bytes,
  };
  int rc = store_read_many(c->store, &sr, 1);
  strbuf_free(&key);
  if (rc) {
    free(footer);
    return DAMACY_IO;
  }

  struct zarr_shard_entry* entries = (struct zarr_shard_entry*)calloc(
    (size_t)n_inner, sizeof(struct zarr_shard_entry));
  if (!entries) {
    free(footer);
    return DAMACY_OOM;
  }
  if (zarr_shard_index_parse(footer, idx_bytes, (size_t)n_inner, entries)) {
    free(footer);
    free(entries);
    return DAMACY_DECODE;
  }
  free(footer);

  struct shard_entry* se = (struct shard_entry*)calloc(1, sizeof(*se));
  if (!se) {
    free(entries);
    return DAMACY_OOM;
  }
  se->uri = str_dup(uri);
  if (!se->uri) {
    free(se);
    free(entries);
    return DAMACY_OOM;
  }
  se->rank = meta->rank;
  for (uint8_t d = 0; d < meta->rank; ++d)
    se->shard_coord[d] = shard_coord[d];
  se->n_entries = n_inner;
  se->entries = entries;

  le = lru_put(c->lru, h, &probe, se);
  if (!le) {
    // lru_put already destroyed se via shard_destroy.
    return DAMACY_OOM;
  }
  const struct shard_entry* held =
    (const struct shard_entry*)lru_entry_value(le);
  *out_entries = held->entries;
  *out_n_entries = held->n_entries;
  return DAMACY_OK;
}

void
zarr_shard_cache_stats_get(const struct zarr_shard_cache* c,
                           struct zarr_shard_cache_stats* out)
{
  if (!out)
    return;
  if (!c) {
    memset(out, 0, sizeof(*out));
    return;
  }
  struct lru_stats s;
  lru_stats_get(c->lru, &s);
  out->hits = s.hits;
  out->misses = s.misses;
  out->insertions = s.insertions;
  out->evictions = s.evictions;
  out->size = s.size;
  out->capacity = s.capacity;
}
