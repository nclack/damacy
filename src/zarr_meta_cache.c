#include "zarr_meta_cache.h"

#include "store.h"
#include "util/hash.h"
#include "util/lru.h"
#include "util/strbuf.h"
#include "zarr_metadata.h"

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

static int
meta_eq(const void* value, const void* probe_key, void* user)
{
  (void)user;
  const struct meta_entry* e = (const struct meta_entry*)value;
  const char* uri = (const char*)probe_key;
  return strcmp(e->uri, uri) == 0;
}

static void
meta_destroy(void* value, void* user)
{
  (void)user;
  struct meta_entry* e = (struct meta_entry*)value;
  if (!e)
    return;
  free(e->uri);
  free(e);
}

// Build "<uri>/zarr.json" or just "zarr.json" when uri is empty.
static int
join_meta_key(struct strbuf* sb, const char* uri)
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
  return strbuf_append_cstr(sb, "zarr.json");
}

struct zarr_meta_cache*
zarr_meta_cache_create(struct store* store, uint32_t capacity)
{
  if (!store || capacity == 0)
    return NULL;
  struct zarr_meta_cache* c = (struct zarr_meta_cache*)calloc(1, sizeof(*c));
  if (!c)
    return NULL;
  c->store = store;
  struct lru_ops ops = {
    .eq = meta_eq,
    .destroy = meta_destroy,
    .user = NULL,
  };
  // max_probe = 16: with idx_size = 2 * capacity (>= 32) and FNV-1a on
  // distinct strings, observed chain length stays well under 16.
  c->lru = lru_create(capacity, 16, &ops);
  if (!c->lru) {
    free(c);
    return NULL;
  }
  return c;
}

void
zarr_meta_cache_destroy(struct zarr_meta_cache* c)
{
  if (!c)
    return;
  lru_destroy(c->lru);
  free(c);
}

enum damacy_status
zarr_meta_cache_get(struct zarr_meta_cache* c,
                    const char* uri,
                    const struct zarr_metadata** out)
{
  if (!c || !uri || !out)
    return DAMACY_INVAL;
  *out = NULL;

  uint64_t h = hash_fnv1a_str(uri);
  struct lru_entry* e = lru_get(c->lru, h, uri);
  if (e) {
    *out = &((const struct meta_entry*)lru_entry_value(e))->meta;
    return DAMACY_OK;
  }

  struct strbuf key = { 0 };
  if (join_meta_key(&key, uri)) {
    strbuf_free(&key);
    return DAMACY_OOM;
  }

  struct store_view view = { 0 };
  int rc = store_map(c->store, strbuf_cstr(&key), &view);
  strbuf_free(&key);
  if (rc)
    return DAMACY_NOTFOUND;

  struct meta_entry* entry = (struct meta_entry*)calloc(1, sizeof(*entry));
  if (!entry) {
    store_unmap(c->store, &view);
    return DAMACY_OOM;
  }
  if (zarr_metadata_parse((const char*)view.data, view.len, &entry->meta)) {
    store_unmap(c->store, &view);
    free(entry);
    return DAMACY_DECODE;
  }
  store_unmap(c->store, &view);

  entry->uri = str_dup(uri);
  if (!entry->uri) {
    free(entry);
    return DAMACY_OOM;
  }

  e = lru_put(c->lru, h, uri, entry);
  if (!e) {
    // Cache full and all entries pinned; lru_put already destroyed entry.
    return DAMACY_OOM;
  }
  *out = &((const struct meta_entry*)lru_entry_value(e))->meta;
  return DAMACY_OK;
}

void
zarr_meta_cache_stats_get(const struct zarr_meta_cache* c,
                          struct zarr_meta_cache_stats* out)
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
