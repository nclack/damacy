#include "prefetch/chunk_layout.h"

#include "damacy.h"
#include "damacy_limits.h"
#include "log/log.h"
#include "prefetch/prefetch_cache.h"
#include "prefetch/shard_index.h"
#include "store/metadata_store_async.h"
#include "store/store.h"
#include "util/hash.h"
#include "util/prelude.h"
#include "util/strbuf.h"
#include "zarr/zarr_chunk_layout.h"
#include "zarr/zarr_metadata.h"
#include "zarr/zarr_shard_index.h"

#include <stdlib.h>
#include <string.h>

struct chunk_layout_key_owned
{
  char* uri;
  uint8_t rank;
  uint32_t n_shards;
  uint64_t* shard_coords;           // flat [n_shards][rank]
  struct prefetch_handle* h_shards; // [n_shards]
};

static void
chunk_layout_key_destroy(const struct prefetch_ops* self, void* stored_key);

static int
chunk_layout_key_eq(const struct prefetch_ops* self,
                    const void* stored_key,
                    const void* probe_key)
{
  (void)self;
  const struct chunk_layout_key_owned* a =
    (const struct chunk_layout_key_owned*)stored_key;
  const struct chunk_layout_key* b = (const struct chunk_layout_key*)probe_key;
  if (a->rank != b->rank || a->n_shards != b->n_shards)
    return 0;
  if (strcmp(a->uri, b->uri) != 0)
    return 0;
  size_t n_coord = (size_t)a->n_shards * a->rank;
  if (n_coord &&
      memcmp(a->shard_coords, b->shard_coords, n_coord * sizeof(uint64_t)) != 0)
    return 0;
  if (a->n_shards &&
      memcmp(a->h_shards,
             b->h_shards,
             (size_t)a->n_shards * sizeof(struct prefetch_handle)) != 0)
    return 0;
  return 1;
}

static void*
chunk_layout_key_clone(const struct prefetch_ops* self, const void* probe_key)
{
  (void)self;
  const struct chunk_layout_key* pk = (const struct chunk_layout_key*)probe_key;
  struct chunk_layout_key_owned* k =
    (struct chunk_layout_key_owned*)calloc(1, sizeof(*k));
  if (!k)
    return NULL;
  k->uri = strdup(pk->uri);
  if (!k->uri)
    goto Error;
  k->rank = pk->rank;
  k->n_shards = pk->n_shards;
  if (pk->n_shards) {
    size_t n_coord = (size_t)pk->n_shards * pk->rank;
    k->shard_coords = (uint64_t*)malloc(n_coord * sizeof(uint64_t));
    k->h_shards = (struct prefetch_handle*)malloc((size_t)pk->n_shards *
                                                  sizeof(*k->h_shards));
    if (!k->shard_coords || !k->h_shards)
      goto Error;
    memcpy(k->shard_coords, pk->shard_coords, n_coord * sizeof(uint64_t));
    memcpy(k->h_shards,
           pk->h_shards,
           (size_t)pk->n_shards * sizeof(struct prefetch_handle));
  }
  return k;

Error:
  chunk_layout_key_destroy(self, k);
  return NULL;
}

static void
chunk_layout_key_destroy(const struct prefetch_ops* self, void* stored_key)
{
  (void)self;
  struct chunk_layout_key_owned* k = (struct chunk_layout_key_owned*)stored_key;
  if (!k)
    return;
  free(k->uri);
  free(k->shard_coords);
  free(k->h_shards);
  free(k);
}

static void
chunk_layout_value_destroy(const struct prefetch_ops* self, void* value)
{
  (void)self;
  free(value);
}

const struct prefetch_ops chunk_layout_ops = {
  .key_eq = chunk_layout_key_eq,
  .key_clone = chunk_layout_key_clone,
  .key_destroy = chunk_layout_key_destroy,
  .value_destroy = chunk_layout_value_destroy,
};

uint64_t
chunk_layout_key_hash(const struct chunk_layout_key* k)
{
  uint64_t h = hash_fnv1a_str(k->uri);
  size_t n_coord = (size_t)k->n_shards * k->rank;
  if (n_coord)
    h =
      hash_combine(h, hash_fnv1a(k->shard_coords, n_coord * sizeof(uint64_t)));
  return h;
}

static int
chunk_layout_fetch(struct prefetch_fetcher* self_,
                   const void* key,
                   void** out_value,
                   int* out_err)
{
  struct chunk_layout_fetcher* self =
    container_of(self_, struct chunk_layout_fetcher, base);
  const struct chunk_layout_key_owned* k =
    (const struct chunk_layout_key_owned*)key;
  const char* uri = k->uri;

  // Borrowed under prefetcher stage ordering; upstream entry remains pinned
  // until this slot transitions out of PENDING.
  const struct zarr_metadata* meta =
    (const struct zarr_metadata*)prefetch_cache_peek(
      self->array_meta_cache, hash_fnv1a_str(uri), uri);
  if (!meta) {
    log_error("chunk_layout_fetch: array meta not ready for uri=%s", uri);
    *out_err = DAMACY_INVAL;
    return 1;
  }
  switch (meta->inner_codec.id) {
    case CODEC_BLOSC_ZSTD:
      break;
    case CODEC_NONE:
    case CODEC_ZSTD:
      // Whole-chunk decode in wave_pool; no blosc1 sub-stream layout to probe.
      *out_value = NULL;
      return 0;
    default:
      log_error("chunk_layout_fetch: unsupported inner codec id=%d (uri=%s)",
                (int)meta->inner_codec.id,
                uri);
      *out_err = DAMACY_DECODE;
      return 1;
  }

  uint64_t chunk_off = 0;
  uint64_t chunk_nbytes = 0;
  const uint64_t* shard_coord = NULL;
  const struct shard_index_value* sv = NULL;
  int found = 0;
  for (uint32_t shard_idx = 0; shard_idx < k->n_shards && !found; ++shard_idx) {
    int shard_err = 0;
    const void* shard_value = NULL;
    enum prefetch_state st = prefetch_cache_query(self->shard_index_cache,
                                                  k->h_shards[shard_idx],
                                                  &shard_value,
                                                  &shard_err);
    if (st == PREFETCH_STATE_ERROR && shard_err == DAMACY_NOTFOUND)
      continue;
    if (st != PREFETCH_STATE_READY) {
      *out_err = shard_err ? shard_err : DAMACY_INVAL;
      return 1;
    }
    sv = (const struct shard_index_value*)shard_value;
    if (!sv)
      continue;
    for (uint64_t i = 0; i < sv->n_entries; ++i) {
      if (sv->entries[i].offset != ZARR_SHARD_EMPTY_OFFSET) {
        chunk_off = sv->entries[i].offset;
        chunk_nbytes = sv->entries[i].nbytes;
        shard_coord = &k->shard_coords[(size_t)shard_idx * k->rank];
        found = 1;
        break;
      }
    }
  }
  if (!found) {
    *out_value = NULL;
    return 0;
  }

  struct strbuf path = { 0 };
  if (zarr_shard_path_build(&path, uri, shard_coord, meta->rank)) {
    strbuf_free(&path);
    *out_err = DAMACY_OOM;
    return 1;
  }

  struct chunk_layout probed = { 0 };
  int rc = zarr_chunk_layout_probe(self->store,
                                   strbuf_cstr(&path),
                                   chunk_off,
                                   (uint32_t)chunk_nbytes,
                                   meta->inner_codec.id,
                                   self->max_substreams_per_chunk,
                                   &probed);
  strbuf_free(&path);
  if (rc) {
    *out_err = DAMACY_DECODE;
    return 1;
  }

  struct chunk_layout* layout = (struct chunk_layout*)malloc(sizeof(*layout));
  if (!layout) {
    *out_err = DAMACY_OOM;
    return 1;
  }
  *layout = probed;
  *out_value = layout;
  return 0;
}

struct chunk_layout_async_ctx
{
  struct prefetch_completion completion;
  uint32_t first_chunk_cbytes;
  uint8_t codec_id;
  uint32_t max_substreams_per_chunk;
};

static void
chunk_layout_header_done(void* user,
                         enum damacy_status status,
                         void* data,
                         size_t len)
{
  struct chunk_layout_async_ctx* ctx = (struct chunk_layout_async_ctx*)user;
  void* value = NULL;
  int err = status;
  if (status != DAMACY_OK)
    goto Done;
  if (len != 16u) {
    err = DAMACY_DECODE;
    goto Done;
  }
  struct chunk_layout probed = { 0 };
  if (zarr_chunk_layout_parse_header(data,
                                     ctx->first_chunk_cbytes,
                                     ctx->codec_id,
                                     ctx->max_substreams_per_chunk,
                                     &probed)) {
    err = DAMACY_DECODE;
    goto Done;
  }
  struct chunk_layout* layout =
    (struct chunk_layout*)malloc(sizeof(struct chunk_layout));
  if (!layout) {
    err = DAMACY_OOM;
    goto Done;
  }
  *layout = probed;
  value = layout;
  err = 0;

Done:
  free(data);
  prefetch_cache_complete(ctx->completion, value, err);
  free(ctx);
}

static int
chunk_layout_async_start(struct prefetch_async_fetcher* self_,
                         const void* key,
                         struct prefetch_completion completion)
{
  struct chunk_layout_async_fetcher* self =
    container_of(self_, struct chunk_layout_async_fetcher, base);
  const struct chunk_layout_key_owned* k =
    (const struct chunk_layout_key_owned*)key;
  const char* uri = k->uri;

  const struct zarr_metadata* meta =
    (const struct zarr_metadata*)prefetch_cache_peek(
      self->array_meta_cache, hash_fnv1a_str(uri), uri);
  if (!meta) {
    log_error("chunk_layout_async_fetch: array meta not ready for uri=%s", uri);
    prefetch_cache_complete(completion, NULL, DAMACY_INVAL);
    return 0;
  }
  switch (meta->inner_codec.id) {
    case CODEC_BLOSC_ZSTD:
      break;
    case CODEC_NONE:
    case CODEC_ZSTD:
      prefetch_cache_complete(completion, NULL, 0);
      return 0;
    default:
      log_error(
        "chunk_layout_async_fetch: unsupported inner codec id=%d (uri=%s)",
        (int)meta->inner_codec.id,
        uri);
      prefetch_cache_complete(completion, NULL, DAMACY_DECODE);
      return 0;
  }

  uint64_t chunk_off = 0;
  uint64_t chunk_nbytes = 0;
  const uint64_t* shard_coord = NULL;
  int found = 0;
  for (uint32_t shard_idx = 0; shard_idx < k->n_shards && !found; ++shard_idx) {
    int shard_err = 0;
    const void* shard_value = NULL;
    enum prefetch_state st = prefetch_cache_query(self->shard_index_cache,
                                                  k->h_shards[shard_idx],
                                                  &shard_value,
                                                  &shard_err);
    if (st == PREFETCH_STATE_ERROR && shard_err == DAMACY_NOTFOUND)
      continue;
    if (st != PREFETCH_STATE_READY) {
      prefetch_cache_complete(
        completion, NULL, shard_err ? shard_err : DAMACY_INVAL);
      return 0;
    }
    const struct shard_index_value* sv =
      (const struct shard_index_value*)shard_value;
    if (!sv)
      continue;
    for (uint64_t i = 0; i < sv->n_entries; ++i) {
      if (sv->entries[i].offset != ZARR_SHARD_EMPTY_OFFSET) {
        chunk_off = sv->entries[i].offset;
        chunk_nbytes = sv->entries[i].nbytes;
        shard_coord = &k->shard_coords[(size_t)shard_idx * k->rank];
        found = 1;
        break;
      }
    }
  }
  if (!found) {
    prefetch_cache_complete(completion, NULL, 0);
    return 0;
  }
  if (chunk_nbytes < 16u) {
    prefetch_cache_complete(completion, NULL, DAMACY_DECODE);
    return 0;
  }

  struct strbuf path = { 0 };
  if (zarr_shard_path_build(&path, uri, shard_coord, meta->rank)) {
    strbuf_free(&path);
    return 1;
  }
  struct chunk_layout_async_ctx* ctx =
    (struct chunk_layout_async_ctx*)malloc(sizeof(*ctx));
  if (!ctx) {
    strbuf_free(&path);
    return 1;
  }
  *ctx = (struct chunk_layout_async_ctx){
    .completion = completion,
    .first_chunk_cbytes = (uint32_t)chunk_nbytes,
    .codec_id = meta->inner_codec.id,
    .max_substreams_per_chunk = self->max_substreams_per_chunk,
  };
  int rc = metadata_store_async_read(self->store,
                                     strbuf_cstr(&path),
                                     chunk_off,
                                     16u,
                                     chunk_layout_header_done,
                                     ctx);
  strbuf_free(&path);
  if (rc) {
    free(ctx);
    return 1;
  }
  return 0;
}

void
chunk_layout_fetcher_init(struct chunk_layout_fetcher* f,
                          struct store* store,
                          struct prefetch_cache* array_meta_cache,
                          struct prefetch_cache* shard_index_cache,
                          uint32_t max_substreams_per_chunk)
{
  CHECK(End, f);
  CHECK(End, store);
  CHECK(End, array_meta_cache);
  CHECK(End, shard_index_cache);
  CHECK(End, max_substreams_per_chunk > 0);
  *f = (struct chunk_layout_fetcher){
    .base = { .fetch = chunk_layout_fetch },
    .store = store,
    .array_meta_cache = array_meta_cache,
    .shard_index_cache = shard_index_cache,
    .max_substreams_per_chunk = max_substreams_per_chunk,
  };
End:
  return;
}

void
chunk_layout_async_fetcher_init(struct chunk_layout_async_fetcher* f,
                                struct metadata_store_async* store,
                                struct prefetch_cache* array_meta_cache,
                                struct prefetch_cache* shard_index_cache,
                                uint32_t max_substreams_per_chunk)
{
  CHECK(End, f);
  CHECK(End, store);
  CHECK(End, array_meta_cache);
  CHECK(End, shard_index_cache);
  CHECK(End, max_substreams_per_chunk > 0);
  *f = (struct chunk_layout_async_fetcher){
    .base = { .start = chunk_layout_async_start },
    .store = store,
    .array_meta_cache = array_meta_cache,
    .shard_index_cache = shard_index_cache,
    .max_substreams_per_chunk = max_substreams_per_chunk,
  };
End:
  return;
}
