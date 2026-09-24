#include "ngff/ngff.h"

#include "pipeline/components.h"
#include "platform/platform.h"
#include "store/metadata_store_async.h"

#include <stdlib.h>
#include <string.h>

struct metadata_read
{
  struct platform_mutex* mutex;
  struct platform_cond* cond;
  int done;
  enum damacy_status status;
  void* data;
  size_t bytes;
};

static void
read_complete(void* user, enum damacy_status status, void* data, size_t bytes)
{
  struct metadata_read* read = user;
  platform_mutex_lock(read->mutex);
  read->status = status;
  read->data = data;
  read->bytes = bytes;
  read->done = 1;
  platform_cond_broadcast(read->cond);
  platform_mutex_unlock(read->mutex);
}

static enum damacy_status
read_metadata(struct metadata_store_async* store,
              struct metadata_read* read,
              const char* uri,
              size_t* remaining_bytes)
{
  size_t len = strlen(uri);
  if (len > SIZE_MAX - sizeof("/zarr.json"))
    return DAMACY_BUDGET;
  char* path = malloc(len + sizeof("/zarr.json"));
  if (!path)
    return DAMACY_OOM;
  memcpy(path, uri, len);
  if (len && path[len - 1] == '/')
    --len;
  memcpy(path + len, "/zarr.json", sizeof("/zarr.json"));
  read->done = 0;
  if (metadata_store_async_read_file_bounded(
        store, path, *remaining_bytes, read_complete, read)) {
    free(path);
    return DAMACY_IO;
  }
  free(path);
  platform_mutex_lock(read->mutex);
  while (!read->done)
    platform_cond_wait(read->cond, read->mutex);
  platform_mutex_unlock(read->mutex);
  if (read->status == DAMACY_OK) {
    if (!read->data || !read->bytes)
      return DAMACY_INVAL;
    *remaining_bytes -= read->bytes;
  }
  return read->status;
}

enum damacy_status
damacy_ngff_image_load(struct damacy_metadata_reader* reader,
                       const char* uri,
                       uint32_t multiscale_index,
                       const struct damacy_ngff_limits* limits,
                       struct damacy_ngff_image** out)
{
  if (!out)
    return DAMACY_INVAL;
  *out = NULL;
  if (!reader || !uri || !*uri || !limits || !limits->max_levels ||
      limits->max_levels > INT32_MAX || !limits->max_metadata_bytes ||
      limits->max_metadata_bytes > SIZE_MAX)
    return DAMACY_INVAL;
  struct metadata_store_async* store = metadata_store_async_create(
    (int)reader->concurrency, NULL, &reader->latency);
  struct metadata_read read = { .mutex = platform_mutex_new(),
                                .cond = platform_cond_new() };
  struct damacy_ngff_image* image = NULL;
  enum damacy_status status = DAMACY_OOM;
  if (!store || !read.mutex || !read.cond)
    goto Done;
  size_t remaining_bytes = (size_t)limits->max_metadata_bytes;
  status = read_metadata(store, &read, uri, &remaining_bytes);
  if (status != DAMACY_OK)
    goto Done;
  status = ngff_parse_group(
    (struct cslice){ read.data, (const char*)read.data + read.bytes },
    uri,
    multiscale_index,
    limits->max_levels,
    &image);
  free(read.data);
  read.data = NULL;
  if (status != DAMACY_OK)
    goto Done;
  for (uint32_t i = 0; i < image->info.level_count; ++i) {
    status =
      read_metadata(store, &read, image->levels[i].uri, &remaining_bytes);
    if (status != DAMACY_OK)
      goto Done;
    status = ngff_parse_array(
      (struct cslice){ read.data, (const char*)read.data + read.bytes },
      image,
      i);
    free(read.data);
    read.data = NULL;
    if (status != DAMACY_OK)
      goto Done;
  }
  *out = image;
  image = NULL;
Done:
  metadata_store_async_destroy(store);
  platform_mutex_free(read.mutex);
  platform_cond_free(read.cond);
  damacy_ngff_image_destroy(image);
  free(read.data);
  return status;
}
