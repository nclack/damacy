#include "zarr/zarr_meta_cache.h"

#include "log/log.h"
#include "platform/platform.h"
#include "store/store.h"
#include "util/atomic_max.h"
#include "util/hash.h"
#include "util/lru.h"
#include "util/prelude.h"
#include "util/strbuf.h"
#include "zarr/zarr_metadata.h"

#include <stdlib.h>
#include <string.h>

struct meta_entry
{
  char* uri;
  struct zarr_metadata meta;
  struct chunk_layout layout;
  uint8_t layout_probed;
};

struct zarr_meta_cache
{
  struct store* store;
  struct lru* lru;
  struct platform_mutex* cache_lock;
  _Atomic(uint16_t)* blosc_nblocks_observer;
};

static int
meta_eq(const void* value, const void* probe_key, void* user)
{
  (void)user;
  const struct meta_entry* entry = (const struct meta_entry*)value;
  return strcmp(entry->uri, (const char*)probe_key) == 0;
}

static void
meta_destroy(void* value, void* user)
{
  (void)user;
  struct meta_entry* entry = (struct meta_entry*)value;
  if (!entry)
    return;
  free(entry->uri);
  free(entry);
}

struct zarr_meta_cache*
zarr_meta_cache_create(struct store* store, uint32_t capacity)
{
  struct zarr_meta_cache* self = NULL;

  CHECK_SILENT(Error, store);
  CHECK_SILENT(Error, capacity > 0);

  self = (struct zarr_meta_cache*)calloc(1, sizeof(*self));
  CHECK(Error, self);
  self->store = store;
  self->cache_lock = platform_mutex_new();
  CHECK(Error, self->cache_lock);

  struct lru_ops ops = {
    .eq = meta_eq,
    .destroy = meta_destroy,
  };
  self->lru = lru_create(capacity, 16, &ops);
  CHECK(Error, self->lru);

  return self;

Error:
  zarr_meta_cache_destroy(self);
  return NULL;
}

void
zarr_meta_cache_destroy(struct zarr_meta_cache* self)
{
  if (!self)
    return;
  lru_destroy(self->lru);
  platform_mutex_free(self->cache_lock);
  free(self);
}

void
zarr_meta_cache_set_blosc_nblocks_observer(struct zarr_meta_cache* self,
                                           _Atomic(uint16_t)* observer)
{
  if (!self)
    return;
  self->blosc_nblocks_observer = observer;
}

enum damacy_status
zarr_meta_cache_get(struct zarr_meta_cache* self,
                    const char* uri,
                    struct zarr_metadata* out)
{
  CHECK_SILENT(Invalid, self);
  CHECK_SILENT(Invalid, uri);
  CHECK_SILENT(Invalid, out);

  uint64_t hash = hash_fnv1a_str(uri);
  platform_mutex_lock(self->cache_lock);
  struct lru_entry* hit = lru_get(self->lru, hash, uri);
  if (hit) {
    *out = ((const struct meta_entry*)lru_entry_value(hit))->meta;
    platform_mutex_unlock(self->cache_lock);
    return DAMACY_OK;
  }
  platform_mutex_unlock(self->cache_lock);

  struct strbuf key = { 0 };
  if (strbuf_join_path(&key, uri, "zarr.json")) {
    strbuf_free(&key);
    return DAMACY_OOM;
  }

  struct store_view view = { 0 };
  int rc = store_map(self->store, strbuf_cstr(&key), &view);
  strbuf_free(&key);
  if (rc)
    return DAMACY_NOTFOUND;

  struct meta_entry* entry = (struct meta_entry*)calloc(1, sizeof(*entry));
  if (!entry) {
    store_unmap(self->store, &view);
    return DAMACY_OOM;
  }
  if (zarr_metadata_parse((const char*)view.data, view.len, &entry->meta)) {
    store_unmap(self->store, &view);
    free(entry);
    return DAMACY_DECODE;
  }
  store_unmap(self->store, &view);

  entry->uri = strdup(uri);
  if (!entry->uri) {
    free(entry);
    return DAMACY_OOM;
  }

  platform_mutex_lock(self->cache_lock);
  // lru_peek doesn't count as a hit/miss and doesn't promote MRU.
  struct lru_entry* existing = lru_peek(self->lru, hash, uri);
  if (existing) {
    *out = ((const struct meta_entry*)lru_entry_value(existing))->meta;
    platform_mutex_unlock(self->cache_lock);
    meta_destroy(entry, NULL);
    return DAMACY_OK;
  }
  struct lru_entry* inserted = lru_put(self->lru, hash, uri, entry);
  if (!inserted) {
    platform_mutex_unlock(self->cache_lock);
    return DAMACY_OOM;
  }
  *out = ((const struct meta_entry*)lru_entry_value(inserted))->meta;
  platform_mutex_unlock(self->cache_lock);
  return DAMACY_OK;

Invalid:
  if (out)
    memset(out, 0, sizeof *out);
  return DAMACY_INVAL;
}

int
zarr_meta_cache_layout_get(struct zarr_meta_cache* self,
                           const char* uri,
                           struct chunk_layout* out)
{
  if (!self || !uri || !out)
    return 1;
  memset(out, 0, sizeof *out);
  uint64_t hash = hash_fnv1a_str(uri);
  platform_mutex_lock(self->cache_lock);
  struct lru_entry* hit = lru_get(self->lru, hash, uri);
  int rc = 1;
  if (hit) {
    struct meta_entry* entry = (struct meta_entry*)lru_entry_value(hit);
    if (entry->layout_probed) {
      *out = entry->layout;
      rc = 0;
    }
  }
  platform_mutex_unlock(self->cache_lock);
  return rc;
}

int
zarr_meta_cache_layout_set(struct zarr_meta_cache* self,
                           const char* uri,
                           const struct chunk_layout* layout)
{
  if (!self || !uri || !layout)
    return 1;
  uint64_t hash = hash_fnv1a_str(uri);
  platform_mutex_lock(self->cache_lock);
  struct lru_entry* hit = lru_get(self->lru, hash, uri);
  if (!hit) {
    platform_mutex_unlock(self->cache_lock);
    return 1;
  }
  struct meta_entry* entry = (struct meta_entry*)lru_entry_value(hit);
  if (!entry->layout_probed) {
    CHECK(Unlock, layout->nblocks <= DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK);
    entry->layout = *layout;
    entry->layout_probed = 1;
    atomic_u16_observe_max(self->blosc_nblocks_observer,
                           (uint16_t)layout->nblocks);
  }
  platform_mutex_unlock(self->cache_lock);
  return 0;
Unlock:
  platform_mutex_unlock(self->cache_lock);
  return 1;
}

int
zarr_meta_cache_probe_layout(struct zarr_meta_cache* self,
                             const char* uri,
                             const char* shard_path,
                             uint64_t first_chunk_off,
                             uint32_t first_chunk_cbytes,
                             uint8_t codec_id,
                             struct chunk_layout* out)
{
  if (!self || !uri || !shard_path || !out)
    return 1;
  uint64_t hash = hash_fnv1a_str(uri);
  platform_mutex_lock(self->cache_lock);
  struct lru_entry* hit = lru_get(self->lru, hash, uri);
  if (!hit) {
    platform_mutex_unlock(self->cache_lock);
    return 1;
  }
  const struct meta_entry* entry =
    (const struct meta_entry*)lru_entry_value(hit);
  if (entry->layout_probed) {
    *out = entry->layout;
    platform_mutex_unlock(self->cache_lock);
    return 0;
  }
  platform_mutex_unlock(self->cache_lock);

  struct chunk_layout probed = { 0 };
  if (zarr_chunk_layout_probe(self->store,
                              shard_path,
                              first_chunk_off,
                              first_chunk_cbytes,
                              codec_id,
                              &probed))
    return 1;

  platform_mutex_lock(self->cache_lock);
  hit = lru_peek(self->lru, hash, uri);
  if (!hit) {
    platform_mutex_unlock(self->cache_lock);
    return 1;
  }
  struct meta_entry* fresh = (struct meta_entry*)lru_entry_value(hit);
  if (!fresh->layout_probed) {
    CHECK(Unlock, probed.nblocks <= DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK);
    fresh->layout = probed;
    fresh->layout_probed = 1;
    atomic_u16_observe_max(self->blosc_nblocks_observer,
                           (uint16_t)probed.nblocks);
  }
  *out = fresh->layout;
  platform_mutex_unlock(self->cache_lock);
  return 0;
Unlock:
  platform_mutex_unlock(self->cache_lock);
  return 1;
}

void
zarr_meta_cache_stats_get(const struct zarr_meta_cache* self,
                          struct zarr_meta_cache_stats* out)
{
  if (!out)
    return;
  struct lru_stats stats;
  if (self)
    platform_mutex_lock(self->cache_lock);
  lru_stats_get(self ? self->lru : NULL, &stats);
  if (self)
    platform_mutex_unlock(self->cache_lock);
  *out = (struct zarr_meta_cache_stats){
    .counters = stats.counters,
    .size = stats.size,
    .capacity = stats.capacity,
  };
}
