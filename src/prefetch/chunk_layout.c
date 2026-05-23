#include "prefetch/chunk_layout.h"

#include "damacy.h"
#include "damacy_limits.h"
#include "log/log.h"
#include "prefetch/prefetch_cache.h"
#include "prefetch/shard_index.h"
#include "store/store.h"
#include "util/hash.h"
#include "util/prelude.h"
#include "util/strbuf.h"
#include "zarr/zarr_chunk_layout.h"
#include "zarr/zarr_metadata.h"
#include "zarr/zarr_shard_index.h"

#include <stdlib.h>
#include <string.h>

static int
chunk_layout_key_eq(const struct prefetch_ops* self,
                    const void* stored_key,
                    const void* probe_key)
{
  (void)self;
  return strcmp((const char*)stored_key, (const char*)probe_key) == 0;
}

static void*
chunk_layout_key_clone(const struct prefetch_ops* self, const void* probe_key)
{
  (void)self;
  return strdup((const char*)probe_key);
}

static void
chunk_layout_key_destroy(const struct prefetch_ops* self, void* stored_key)
{
  (void)self;
  free(stored_key);
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

static int
chunk_layout_fetch(struct prefetch_fetcher* self_,
                   const void* key,
                   void** out_value,
                   int* out_err)
{
  struct chunk_layout_fetcher* self =
    container_of(self_, struct chunk_layout_fetcher, base);
  const char* uri = (const char*)key;

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
  if (meta->inner_codec.id != CODEC_BLOSC_ZSTD) {
    // Probe is blosc1-specific; non-blosc arrays carry no chunk layout
    // and the decoder uses worst-case caps. Surface as success with no
    // value so the sample's prefetch stage reaches READY.
    *out_value = NULL;
    return 0;
  }

  // TODO(perf): origin-shard probe; far-from-origin workloads silently
  // fall through to worst-case decoder caps. Needs cache API extension.
  struct shard_index_key probe = { .uri = uri, .rank = meta->rank };
  for (uint8_t d = 0; d < meta->rank; ++d)
    probe.shard_coord[d] = 0;

  // Borrowed under prefetcher stage ordering; upstream entry remains pinned
  // until this slot transitions out of PENDING.
  const struct shard_index_value* sv =
    (const struct shard_index_value*)prefetch_cache_peek(
      self->shard_index_cache, shard_index_key_hash(&probe), &probe);
  if (!sv) {
    log_error("chunk_layout_fetch: shard_index not ready for uri=%s", uri);
    *out_err = DAMACY_INVAL;
    return 1;
  }

  uint64_t chunk_off = 0;
  uint64_t chunk_nbytes = 0;
  int found = 0;
  for (uint64_t i = 0; i < sv->n_entries; ++i) {
    if (sv->entries[i].offset != ZARR_SHARD_EMPTY_OFFSET) {
      chunk_off = sv->entries[i].offset;
      chunk_nbytes = sv->entries[i].nbytes;
      found = 1;
      break;
    }
  }
  if (!found) {
    *out_err = DAMACY_NOTFOUND;
    return 1;
  }

  struct strbuf path = { 0 };
  if (zarr_shard_path_build(&path, uri, probe.shard_coord, meta->rank)) {
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
