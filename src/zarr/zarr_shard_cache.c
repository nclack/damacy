#include "zarr/zarr_shard_cache.h"

#include "damacy_limits.h"
#include "log/log.h"
#include "platform/platform.h"
#include "store/store.h"
#include "util/hash.h"
#include "util/lru.h"
#include "util/path_intern.h"
#include "util/prelude.h"
#include "util/strbuf.h"
#include "zarr/zarr_metadata.h"
#include "zarr/zarr_shard_index.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

struct shard_entry
{
  const char* uri; // interned; not owned
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
  struct store* store;      // borrowed
  struct path_intern* uris; // borrowed; may be NULL
  struct lru* lru;
  struct platform_mutex* mu; // guards lru_get / lru_put / lru_stats_get
};

static uint64_t
shard_hash(const char* uri, const uint64_t* shard_coord, uint8_t rank)
{
  uint64_t uri_hash = hash_ptr(uri);
  uint64_t coord_hash =
    hash_fnv1a(shard_coord, (size_t)rank * sizeof(uint64_t));
  return hash_combine(uri_hash, coord_hash);
}

static int
shard_eq(const void* value, const void* probe_key, void* user)
{
  (void)user;
  const struct shard_entry* entry = (const struct shard_entry*)value;
  const struct shard_probe* probe = (const struct shard_probe*)probe_key;
  if (entry->rank != probe->rank)
    return 0;
  if (entry->uri != probe->uri)
    return 0;
  for (uint8_t d = 0; d < entry->rank; ++d)
    if (entry->shard_coord[d] != probe->shard_coord[d])
      return 0;
  return 1;
}

static void
shard_destroy(void* value, void* user)
{
  struct shard_entry* entry = (struct shard_entry*)value;
  if (!entry)
    return;
  struct zarr_shard_cache* cache = (struct zarr_shard_cache*)user;
  if (cache && cache->uris && entry->uri)
    path_intern_release(cache->uris, entry->uri);
  free(entry->entries);
  free(entry);
}

struct zarr_shard_cache*
zarr_shard_cache_create(struct store* store,
                        struct path_intern* uris,
                        uint32_t capacity)
{
  struct zarr_shard_cache* self = NULL;

  CHECK_SILENT(Error, store);
  CHECK_SILENT(Error, capacity > 0);

  self = (struct zarr_shard_cache*)calloc(1, sizeof(*self));
  CHECK(Error, self);
  self->store = store;
  self->uris = uris;

  struct lru_ops ops = {
    .eq = shard_eq,
    .destroy = shard_destroy,
    .user = self,
  };
  // hash_combine over (hash_ptr(uri), hash_fnv1a(shard_coord)) — the
  // multiplicative URI hash and content coord hash mix well; observed
  // chain depths stay well under the cap.
  self->lru = lru_create(capacity, 16, &ops);
  CHECK(Error, self->lru);
  self->mu = platform_mutex_new();
  CHECK(Error, self->mu);

  return self;

Error:
  zarr_shard_cache_destroy(self);
  return NULL;
}

void
zarr_shard_cache_destroy(struct zarr_shard_cache* self)
{
  if (!self)
    return;
  platform_mutex_free(self->mu);
  lru_destroy(self->lru);
  free(self);
}

enum damacy_status
zarr_shard_cache_get(struct zarr_shard_cache* self,
                     const char* uri,
                     const struct zarr_metadata* meta,
                     const uint64_t* shard_coord,
                     struct zarr_shard_pin* out_pin,
                     const struct zarr_shard_entry** out_entries,
                     uint64_t* out_n_entries)
{
  CHECK_SILENT(Invalid, self);
  CHECK_SILENT(Invalid, uri);
  CHECK_SILENT(Invalid, meta);
  CHECK_SILENT(Invalid, shard_coord);
  CHECK_SILENT(Invalid, out_pin);
  CHECK_SILENT(Invalid, out_entries);
  CHECK_SILENT(Invalid, out_n_entries);
  assert((!self->uris || path_intern_owns(self->uris, uri)) &&
         "zarr_shard_cache_get: uri must be a path_intern pointer");
  out_pin->opaque = NULL;
  if (meta->rank == 0 || meta->rank > DAMACY_MAX_RANK) {
    *out_entries = NULL;
    *out_n_entries = 0;
    return DAMACY_RANK;
  }
  *out_entries = NULL;
  *out_n_entries = 0;

  struct shard_probe probe = {
    .uri = uri,
    .rank = meta->rank,
    .shard_coord = shard_coord,
  };
  uint64_t hash = shard_hash(uri, shard_coord, meta->rank);

  // Acquire the pin under the cache mutex so the refcount bump is
  // atomic with the lookup — eviction selection won't touch a slot
  // whose refcount > 0, so once we hold the pin the (entries,
  // n_entries) pair stays valid until release.
  platform_mutex_lock(self->mu);
  struct lru_entry* hit = lru_get(self->lru, hash, &probe);
  if (hit) {
    lru_entry_acquire(hit);
    const struct shard_entry* entry =
      (const struct shard_entry*)lru_entry_value(hit);
    *out_entries = entry->entries;
    *out_n_entries = entry->n_entries;
    out_pin->opaque = hit;
    platform_mutex_unlock(self->mu);
    return DAMACY_OK;
  }
  platform_mutex_unlock(self->mu);

  uint64_t n_inner_per_shard = 0;
  if (zarr_metadata_inner_per_shard(meta, NULL, &n_inner_per_shard))
    return DAMACY_INVAL;

  struct strbuf key = { 0 };
  if (zarr_shard_path_build(&key, uri, shard_coord, meta->rank)) {
    strbuf_free(&key);
    return DAMACY_OOM;
  }

  // Missing or zero-length file → NOTFOUND. Upstream routes that to the
  // array's fill_value (zarr v3 spec: absent chunks read as fill).
  // Zero-byte non-sharded files aren't standard but are unambiguous; for
  // sharded arrays a zero-byte file is corrupt (no index could fit) and
  // upstream's fill substitution is wrong-but-safe — flag in followup.
  uint64_t file_n_bytes = 0;
  if (store_stat(self->store, strbuf_cstr(&key), &file_n_bytes) ||
      file_n_bytes == 0) {
    strbuf_free(&key);
    return DAMACY_NOTFOUND;
  }

  struct zarr_shard_entry* entries = (struct zarr_shard_entry*)calloc(
    (size_t)n_inner_per_shard, sizeof(struct zarr_shard_entry));
  if (!entries) {
    strbuf_free(&key);
    return DAMACY_OOM;
  }

  if (!meta->sharded) {
    // Non-sharded: the file IS the single inner chunk. Synthesize a
    // 1-entry index pointing at the whole file.
    entries[0].offset = 0;
    entries[0].nbytes = file_n_bytes;
    strbuf_free(&key);
  } else {
    size_t footer_n_bytes = zarr_shard_index_size((size_t)n_inner_per_shard);
    if (file_n_bytes < (uint64_t)footer_n_bytes) {
      strbuf_free(&key);
      free(entries);
      return DAMACY_DECODE;
    }
    uint64_t footer_offset =
      meta->index_location_end ? (file_n_bytes - (uint64_t)footer_n_bytes) : 0;

    void* footer_buf = malloc(footer_n_bytes);
    if (!footer_buf) {
      strbuf_free(&key);
      free(entries);
      return DAMACY_OOM;
    }
    struct store_read read_request = {
      .key = strbuf_cstr(&key),
      .dst = footer_buf,
      .offset = footer_offset,
      .len = footer_n_bytes,
    };
    int rc = store_read_many(self->store, &read_request, 1);
    strbuf_free(&key);
    if (rc) {
      free(footer_buf);
      free(entries);
      return DAMACY_IO;
    }
    if (zarr_shard_index_parse(
          footer_buf, footer_n_bytes, (size_t)n_inner_per_shard, entries)) {
      free(footer_buf);
      free(entries);
      return DAMACY_DECODE;
    }
    free(footer_buf);
  }

  struct shard_entry* entry = (struct shard_entry*)calloc(1, sizeof(*entry));
  if (!entry) {
    free(entries);
    return DAMACY_OOM;
  }
  // Pin the key: without this, the entry survives every caller
  // dropping refs and dangles on the freed string.
  if (self->uris) {
    entry->uri = path_intern_acquire(self->uris, uri);
    if (!entry->uri) {
      free(entry);
      free(entries);
      return DAMACY_OOM;
    }
  } else {
    entry->uri = uri;
  }
  entry->rank = meta->rank;
  for (uint8_t d = 0; d < meta->rank; ++d)
    entry->shard_coord[d] = shard_coord[d];
  entry->n_entries = n_inner_per_shard;
  entry->entries = entries;

  platform_mutex_lock(self->mu);
  // Re-check under the lock: another thread may have raced us on the
  // same probe key while we were doing the unlocked I/O. If so, drop
  // our duplicate and return the winner. lru_peek doesn't count as a
  // miss/hit and doesn't promote.
  struct lru_entry* existing = lru_peek(self->lru, hash, &probe);
  if (existing) {
    shard_destroy(entry, self);
    lru_entry_acquire(existing);
    const struct shard_entry* held =
      (const struct shard_entry*)lru_entry_value(existing);
    *out_entries = held->entries;
    *out_n_entries = held->n_entries;
    out_pin->opaque = existing;
    platform_mutex_unlock(self->mu);
    return DAMACY_OK;
  }
  struct lru_entry* inserted = lru_put(self->lru, hash, &probe, entry);
  if (!inserted) {
    // lru_put already destroyed entry via shard_destroy.
    platform_mutex_unlock(self->mu);
    return DAMACY_OOM;
  }
  lru_entry_acquire(inserted);
  const struct shard_entry* held =
    (const struct shard_entry*)lru_entry_value(inserted);
  *out_entries = held->entries;
  *out_n_entries = held->n_entries;
  out_pin->opaque = inserted;
  platform_mutex_unlock(self->mu);
  return DAMACY_OK;

Invalid:
  if (out_pin)
    out_pin->opaque = NULL;
  if (out_entries)
    *out_entries = NULL;
  if (out_n_entries)
    *out_n_entries = 0;
  return DAMACY_INVAL;
}

void
zarr_shard_cache_release(struct zarr_shard_cache* self,
                         struct zarr_shard_pin pin)
{
  CHECK(Fail, self);
  if (!pin.opaque)
    return;
  platform_mutex_lock(self->mu);
  lru_entry_release((struct lru_entry*)pin.opaque);
  platform_mutex_unlock(self->mu);
  return;
Fail:
  return;
}

void
zarr_shard_cache_stats_get(struct zarr_shard_cache* self,
                           struct zarr_shard_cache_stats* out)
{
  if (!out)
    return;
  struct lru_stats stats;
  if (self)
    platform_mutex_lock(self->mu);
  lru_stats_get(self ? self->lru : NULL, &stats);
  if (self)
    platform_mutex_unlock(self->mu);
  *out = (struct zarr_shard_cache_stats){
    .counters = stats.counters,
    .size = stats.size,
    .capacity = stats.capacity,
  };
}
