#include "zarr/zarr_meta_cache.h"

#include "log/log.h"
#include "store/store.h"
#include "util/hash.h"
#include "util/lru.h"
#include "util/prelude.h"
#include "util/strbuf.h"
#include "zarr/zarr_metadata.h"

#include <stdlib.h>
#include <string.h>

struct meta_entry
{
  char* uri; // owned, NUL-terminated
  struct zarr_metadata meta;
};

struct zarr_meta_cache
{
  struct store* store; // borrowed
  struct lru* lru;
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
  struct lru_entry* hit = lru_get(self->lru, hash, uri);
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

  struct lru_entry* inserted = lru_put(self->lru, hash, uri, entry);
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

void
zarr_meta_cache_stats_get(const struct zarr_meta_cache* self,
                          struct zarr_meta_cache_stats* out)
{
  if (!out)
    return;
  struct lru_stats stats;
  lru_stats_get(self ? self->lru : NULL, &stats);
  *out = (struct zarr_meta_cache_stats){
    .counters = stats.counters,
    .size = stats.size,
    .capacity = stats.capacity,
  };
}
