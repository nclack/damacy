#include "prefetch/array_meta.h"

#include "damacy.h"
#include "log/log.h"
#include "store/metadata_store_async.h"
#include "store/store.h"
#include "util/prelude.h"
#include "util/strbuf.h"
#include "zarr/zarr_metadata.h"

#include <stdlib.h>
#include <string.h>

static int
array_meta_key_eq(const struct prefetch_ops* self,
                  const void* stored_key,
                  const void* probe_key)
{
  (void)self;
  return strcmp((const char*)stored_key, (const char*)probe_key) == 0;
}

static void*
array_meta_key_clone(const struct prefetch_ops* self, const void* probe_key)
{
  (void)self;
  return strdup((const char*)probe_key);
}

static void
array_meta_key_destroy(const struct prefetch_ops* self, void* stored_key)
{
  (void)self;
  free(stored_key);
}

static void
array_meta_value_destroy(const struct prefetch_ops* self, void* value)
{
  (void)self;
  free(value);
}

const struct prefetch_ops array_meta_ops = {
  .key_eq = array_meta_key_eq,
  .key_clone = array_meta_key_clone,
  .key_destroy = array_meta_key_destroy,
  .value_destroy = array_meta_value_destroy,
};

static int
array_meta_fetch(struct prefetch_fetcher* self_,
                 const void* key,
                 void** out_value,
                 int* out_err)
{
  struct array_meta_fetcher* self =
    container_of(self_, struct array_meta_fetcher, base);
  const char* uri = (const char*)key;

  struct strbuf path = { 0 };
  if (strbuf_join_path(&path, uri, "zarr.json")) {
    strbuf_free(&path);
    *out_err = DAMACY_OOM;
    return 1;
  }

  struct store_view view = { 0 };
  int rc = store_map(self->store, strbuf_cstr(&path), &view);
  strbuf_free(&path);
  if (rc) {
    *out_err = DAMACY_NOTFOUND;
    return 1;
  }

  struct zarr_metadata* meta = (struct zarr_metadata*)malloc(sizeof(*meta));
  if (!meta) {
    store_unmap(self->store, &view);
    *out_err = DAMACY_OOM;
    return 1;
  }

  if (zarr_metadata_parse((const char*)view.data, view.len, meta)) {
    store_unmap(self->store, &view);
    free(meta);
    *out_err = DAMACY_DECODE;
    return 1;
  }

  store_unmap(self->store, &view);
  *out_value = meta;
  return 0;
}

struct array_meta_async_ctx
{
  struct prefetch_completion completion;
};

static void
array_meta_read_done(void* user,
                     enum damacy_status status,
                     void* data,
                     size_t len)
{
  struct array_meta_async_ctx* ctx = (struct array_meta_async_ctx*)user;
  void* value = NULL;
  int err = status;
  if (status != DAMACY_OK)
    goto Done;

  struct zarr_metadata* meta = (struct zarr_metadata*)malloc(sizeof(*meta));
  if (!meta) {
    err = DAMACY_OOM;
    goto Done;
  }
  if (zarr_metadata_parse((const char*)data, len, meta)) {
    free(meta);
    err = DAMACY_DECODE;
    goto Done;
  }

  value = meta;
  err = 0;

Done:
  free(data);
  prefetch_cache_complete(ctx->completion, value, err);
  free(ctx);
}

static int
array_meta_async_start(struct prefetch_async_fetcher* self_,
                       const void* key,
                       struct prefetch_completion completion)
{
  struct array_meta_async_fetcher* self =
    container_of(self_, struct array_meta_async_fetcher, base);
  const char* uri = (const char*)key;
  struct strbuf path = { 0 };
  if (strbuf_join_path(&path, uri, "zarr.json")) {
    strbuf_free(&path);
    return 1;
  }
  struct array_meta_async_ctx* ctx =
    (struct array_meta_async_ctx*)malloc(sizeof(*ctx));
  if (!ctx) {
    strbuf_free(&path);
    return 1;
  }
  *ctx = (struct array_meta_async_ctx){ .completion = completion };
  int rc = metadata_store_async_read_file(
    self->store, strbuf_cstr(&path), array_meta_read_done, ctx);
  strbuf_free(&path);
  if (rc) {
    free(ctx);
    return 1;
  }
  return 0;
}

void
array_meta_fetcher_init(struct array_meta_fetcher* f, struct store* store)
{
  CHECK(End, f);
  CHECK(End, store);
  *f = (struct array_meta_fetcher){
    .base = { .fetch = array_meta_fetch },
    .store = store,
  };
End:
  return;
}

void
array_meta_async_fetcher_init(struct array_meta_async_fetcher* f,
                              struct metadata_store_async* store)
{
  CHECK(End, f);
  CHECK(End, store);
  *f = (struct array_meta_async_fetcher){
    .base = { .start = array_meta_async_start },
    .store = store,
  };
End:
  return;
}
