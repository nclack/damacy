#include "prefetch/shard_index.h"

#include "damacy.h"
#include "damacy_limits.h"
#include "log/log.h"
#include "prefetch/prefetch_cache.h"
#include "store/store.h"
#include "util/hash.h"
#include "util/prelude.h"
#include "util/strbuf.h"
#include "zarr/zarr_metadata.h"
#include "zarr/zarr_shard_index.h"

#include <stdlib.h>
#include <string.h>

static int
shard_index_eq(const struct prefetch_ops* self,
               const void* stored_key,
               const void* probe_key)
{
  (void)self;
  const struct shard_index_key* a = (const struct shard_index_key*)stored_key;
  const struct shard_index_key* b = (const struct shard_index_key*)probe_key;
  if (a->rank != b->rank)
    return 0;
  if (strcmp(a->uri, b->uri) != 0)
    return 0;
  for (uint8_t d = 0; d < a->rank; ++d)
    if (a->shard_coord[d] != b->shard_coord[d])
      return 0;
  return 1;
}

static void*
shard_index_key_clone(const struct prefetch_ops* self, const void* probe_key)
{
  (void)self;
  const struct shard_index_key* pk = (const struct shard_index_key*)probe_key;
  struct shard_index_key* k = (struct shard_index_key*)malloc(sizeof(*k));
  if (!k)
    return NULL;
  char* uri_copy = strdup(pk->uri);
  if (!uri_copy) {
    free(k);
    return NULL;
  }
  *k = (struct shard_index_key){
    .uri = uri_copy,
    .rank = pk->rank,
  };
  for (uint8_t d = 0; d < pk->rank; ++d)
    k->shard_coord[d] = pk->shard_coord[d];
  return k;
}

static void
shard_index_key_destroy(const struct prefetch_ops* self, void* stored_key)
{
  (void)self;
  struct shard_index_key* k = (struct shard_index_key*)stored_key;
  if (!k)
    return;
  free((char*)k->uri);
  free(k);
}

static void
shard_index_value_destroy(const struct prefetch_ops* self, void* value)
{
  (void)self;
  struct shard_index_value* v = (struct shard_index_value*)value;
  if (!v)
    return;
  free(v->entries);
  free(v);
}

const struct prefetch_ops shard_index_ops = {
  .key_eq = shard_index_eq,
  .key_clone = shard_index_key_clone,
  .key_destroy = shard_index_key_destroy,
  .value_destroy = shard_index_value_destroy,
};

uint64_t
shard_index_key_hash(const struct shard_index_key* k)
{
  uint64_t uri_h = hash_fnv1a_str(k->uri);
  uint64_t coord_h =
    hash_fnv1a(k->shard_coord, (size_t)k->rank * sizeof(uint64_t));
  return hash_combine(uri_h, coord_h);
}

static int
shard_index_fetch(struct prefetch_fetcher* self_,
                  const void* key,
                  void** out_value,
                  int* out_err)
{
  struct shard_index_fetcher* self =
    container_of(self_, struct shard_index_fetcher, base);
  const struct shard_index_key* k = (const struct shard_index_key*)key;

  // Peek is safe: the prefetcher orders stage-2 after stage-1, so the
  // array-meta entry is still pinned.
  const struct zarr_metadata* meta =
    (const struct zarr_metadata*)prefetch_cache_peek(
      self->array_meta_cache, hash_fnv1a_str(k->uri), k->uri);
  if (!meta) {
    log_error("shard_index_fetch: array meta not ready for uri=%s", k->uri);
    *out_err = DAMACY_INVAL;
    return 1;
  }
  if (meta->rank != k->rank) {
    *out_err = DAMACY_RANK;
    return 1;
  }

  uint64_t n_inner_per_shard = 0;
  if (zarr_metadata_inner_per_shard(meta, NULL, &n_inner_per_shard)) {
    *out_err = DAMACY_INVAL;
    return 1;
  }

  struct strbuf path = { 0 };
  if (zarr_shard_path_build(&path, k->uri, k->shard_coord, k->rank)) {
    strbuf_free(&path);
    *out_err = DAMACY_OOM;
    return 1;
  }

  uint64_t file_n_bytes = 0;
  if (store_stat(self->store, strbuf_cstr(&path), &file_n_bytes) ||
      file_n_bytes == 0) {
    strbuf_free(&path);
    *out_err = DAMACY_NOTFOUND;
    return 1;
  }

  struct zarr_shard_entry* entries = (struct zarr_shard_entry*)calloc(
    (size_t)n_inner_per_shard, sizeof(struct zarr_shard_entry));
  if (!entries) {
    strbuf_free(&path);
    *out_err = DAMACY_OOM;
    return 1;
  }

  if (!meta->sharded) {
    // Non-sharded: the file IS the single inner chunk.
    entries[0].offset = 0;
    entries[0].nbytes = file_n_bytes;
    strbuf_free(&path);
  } else {
    size_t footer_n_bytes = zarr_shard_index_size((size_t)n_inner_per_shard);
    if (file_n_bytes < (uint64_t)footer_n_bytes) {
      strbuf_free(&path);
      free(entries);
      *out_err = DAMACY_DECODE;
      return 1;
    }
    uint64_t footer_offset =
      meta->index_location_end ? (file_n_bytes - (uint64_t)footer_n_bytes) : 0;
    void* footer_buf = malloc(footer_n_bytes);
    if (!footer_buf) {
      strbuf_free(&path);
      free(entries);
      *out_err = DAMACY_OOM;
      return 1;
    }
    struct store_read req = {
      .key = strbuf_cstr(&path),
      .dst = footer_buf,
      .offset = footer_offset,
      .len = footer_n_bytes,
    };
    int rc = store_read_many(self->store, &req, 1);
    strbuf_free(&path);
    if (rc) {
      free(footer_buf);
      free(entries);
      *out_err = DAMACY_IO;
      return 1;
    }
    if (zarr_shard_index_parse(
          footer_buf, footer_n_bytes, (size_t)n_inner_per_shard, entries)) {
      free(footer_buf);
      free(entries);
      *out_err = DAMACY_DECODE;
      return 1;
    }
    free(footer_buf);
  }

  struct shard_index_value* v = (struct shard_index_value*)malloc(sizeof(*v));
  if (!v) {
    free(entries);
    *out_err = DAMACY_OOM;
    return 1;
  }
  *v = (struct shard_index_value){
    .entries = entries,
    .n_entries = n_inner_per_shard,
  };
  *out_value = v;
  return 0;
}

void
shard_index_fetcher_init(struct shard_index_fetcher* f,
                         struct store* store,
                         struct prefetch_cache* array_meta_cache)
{
  CHECK(End, f);
  CHECK(End, store);
  CHECK(End, array_meta_cache);
  *f = (struct shard_index_fetcher){
    .base = { .fetch = shard_index_fetch },
    .store = store,
    .array_meta_cache = array_meta_cache,
  };
End:
  return;
}
