#include "zarr/zarr_meta_cache.h"

#include "log/log.h"
#include "store/store.h"
#include "util/hash.h"
#include "util/lru.h"
#include "util/prelude.h"
#include "util/strbuf.h"
#include "zarr/zarr_metadata.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

struct meta_entry
{
  char* uri; // owned, NUL-terminated
  struct zarr_metadata meta;
  struct chunk_layout layout;
  uint8_t layout_probed;
};

struct zarr_meta_cache
{
  struct store* store; // borrowed
  struct lru* lru;
  pthread_mutex_t mu; // guards layout/layout_probed on cached entries
};

static int
meta_eq(const void* value, const void* probe_key, void* user)
{
  (void)user;
  const struct meta_entry* entry = (const struct meta_entry*)value;
  const char* uri = (const char*)probe_key;
  return strcmp(entry->uri, uri) == 0;
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
  pthread_mutex_init(&self->mu, NULL);

  struct lru_ops ops = {
    .eq = meta_eq,
    .destroy = meta_destroy,
  };
  // max_probe = 16: with n_cells = 2 * capacity (>= 32) and FNV-1a on
  // distinct strings, observed chain length stays well under 16.
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
  pthread_mutex_destroy(&self->mu);
  free(self);
}

enum damacy_status
zarr_meta_cache_get(struct zarr_meta_cache* self,
                    const char* uri,
                    const struct zarr_metadata** out)
{
  CHECK_SILENT(Invalid, self);
  CHECK_SILENT(Invalid, uri);
  CHECK_SILENT(Invalid, out);
  *out = NULL;

  uint64_t hash = hash_fnv1a_str(uri);
  pthread_mutex_lock(&self->mu);
  struct lru_entry* hit = lru_get(self->lru, hash, uri);
  pthread_mutex_unlock(&self->mu);
  if (hit) {
    *out = &((const struct meta_entry*)lru_entry_value(hit))->meta;
    return DAMACY_OK;
  }

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

  pthread_mutex_lock(&self->mu);
  struct lru_entry* inserted = lru_put(self->lru, hash, uri, entry);
  pthread_mutex_unlock(&self->mu);
  if (!inserted) {
    // Cache full and all entries pinned; lru_put already destroyed entry.
    return DAMACY_OOM;
  }
  *out = &((const struct meta_entry*)lru_entry_value(inserted))->meta;
  return DAMACY_OK;

Invalid:
  if (out)
    *out = NULL;
  return DAMACY_INVAL;
}

// layout/layout_probed are guarded by self->mu; the entry pointer itself
// is stable (LRU never reshuffles existing nodes).
const struct chunk_layout*
zarr_meta_cache_layout_get(struct zarr_meta_cache* self, const char* uri)
{
  if (!self || !uri)
    return NULL;
  uint64_t hash = hash_fnv1a_str(uri);
  pthread_mutex_lock(&self->mu);
  struct lru_entry* hit = lru_get(self->lru, hash, uri);
  const struct chunk_layout* out = NULL;
  if (hit) {
    struct meta_entry* entry = (struct meta_entry*)lru_entry_value(hit);
    if (entry->layout_probed)
      out = &entry->layout;
  }
  pthread_mutex_unlock(&self->mu);
  return out;
}

// First writer wins; redundant sets are dropped under the lock.
int
zarr_meta_cache_layout_set(struct zarr_meta_cache* self,
                           const char* uri,
                           const struct chunk_layout* layout)
{
  if (!self || !uri || !layout)
    return 1;
  uint64_t hash = hash_fnv1a_str(uri);
  pthread_mutex_lock(&self->mu);
  struct lru_entry* hit = lru_get(self->lru, hash, uri);
  if (!hit) {
    pthread_mutex_unlock(&self->mu);
    return 1;
  }
  struct meta_entry* entry = (struct meta_entry*)lru_entry_value(hit);
  if (!entry->layout_probed) {
    entry->layout = *layout;
    entry->layout_probed = 1;
  }
  pthread_mutex_unlock(&self->mu);
  return 0;
}

// I/O runs unlocked; commit happens under self->mu. Concurrent probes for
// the same URI read identical bytes, so last-writer-wins is value-safe.
const struct chunk_layout*
zarr_meta_cache_probe_layout(struct zarr_meta_cache* self,
                             const char* uri,
                             const char* shard_path,
                             uint64_t first_chunk_off,
                             uint32_t first_chunk_cbytes,
                             uint8_t codec_id)
{
  if (!self || !uri || !shard_path)
    return NULL;
  uint64_t hash = hash_fnv1a_str(uri);
  pthread_mutex_lock(&self->mu);
  struct lru_entry* hit = lru_get(self->lru, hash, uri);
  if (!hit) {
    pthread_mutex_unlock(&self->mu);
    return NULL;
  }
  struct meta_entry* entry = (struct meta_entry*)lru_entry_value(hit);
  if (entry->layout_probed) {
    const struct chunk_layout* out = &entry->layout;
    pthread_mutex_unlock(&self->mu);
    return out;
  }
  pthread_mutex_unlock(&self->mu);

  struct chunk_layout probed = { 0 };
  if (zarr_chunk_layout_probe(self->store,
                              shard_path,
                              first_chunk_off,
                              first_chunk_cbytes,
                              codec_id,
                              &probed))
    return NULL;

  pthread_mutex_lock(&self->mu);
  if (!entry->layout_probed) {
    entry->layout = probed;
    entry->layout_probed = 1;
  }
  pthread_mutex_unlock(&self->mu);
  return &entry->layout;
}

void
zarr_meta_cache_stats_get(const struct zarr_meta_cache* self,
                          struct zarr_meta_cache_stats* out)
{
  if (!out)
    return;
  struct lru_stats stats;
  if (self)
    pthread_mutex_lock((pthread_mutex_t*)&self->mu);
  lru_stats_get(self ? self->lru : NULL, &stats);
  if (self)
    pthread_mutex_unlock((pthread_mutex_t*)&self->mu);
  *out = (struct zarr_meta_cache_stats){
    .counters = stats.counters,
    .size = stats.size,
    .capacity = stats.capacity,
  };
}
