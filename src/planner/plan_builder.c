#include "planner/plan_builder.h"

#include "damacy_config.h"
#include "prefetch/prefetch_cache.h"
#include "prefetch/shard_index.h"
#include "util/hash.h"
#include "util/strbuf.h"

#include <stdlib.h>
#include <string.h>

static enum damacy_status
array_metadata(struct prefetch_cache* cache,
               struct prefetch_handle handle,
               const struct zarr_metadata** out)
{
  const void* value = NULL;
  int error = 0;
  enum prefetch_state state =
    prefetch_cache_query(cache, handle, &value, &error);
  if (state == PREFETCH_STATE_PENDING)
    return DAMACY_AGAIN;
  if (state == PREFETCH_STATE_ERROR)
    return error ? (enum damacy_status)error : DAMACY_INVAL;
  *out = value;
  return value ? DAMACY_OK : DAMACY_INVAL;
}

static enum damacy_status
sample_geometry(const struct planner_sample* sample,
                const struct zarr_metadata* meta,
                const struct damacy_batch_spec* output,
                const struct damacy_plan_limits* limits,
                uint32_t* count,
                uint32_t* bytes)
{
  if (sample->aabb.rank != meta->rank || meta->rank != output->sample_rank)
    return DAMACY_RANK;
  if (!cast_path_supported(output->dtype, meta->dtype))
    return DAMACY_DTYPE;
  uint64_t size = dtype_bpe(meta->dtype);
  for (uint8_t d = 0; d < meta->rank; ++d) {
    int64_t lo = sample->aabb.dims[d].beg;
    int64_t hi = sample->aabb.dims[d].end;
    uint64_t chunk = meta->inner_chunk_shape[d];
    if (lo < 0 || hi <= lo || (uint64_t)hi > meta->shape[d] || !chunk)
      return DAMACY_INVAL;
    int64_t extent = sample->axes[d].count ? sample->axes[d].count : hi - lo;
    if (extent != output->sample_shape[d])
      return DAMACY_INVAL;
    if (size > limits->max_chunk_bytes / chunk)
      return DAMACY_BUDGET;
    size *= chunk;
  }
  struct selection_grid grid;
  enum damacy_status status = selection_grid_init(&grid,
                                                  &sample->aabb,
                                                  sample->axes,
                                                  meta->inner_chunk_shape,
                                                  NULL,
                                                  NULL,
                                                  limits->max_chunks);
  if (status != DAMACY_OK)
    return status;
  *count = (uint32_t)grid.count;
  *bytes = (uint32_t)size;
  return DAMACY_OK;
}

static uint64_t
chunk_hash(uint32_t array, const uint64_t* coordinate, uint8_t rank)
{
  uint64_t hash = array;
  for (uint8_t d = 0; d < rank; ++d)
    hash = hash_combine(hash, coordinate[d]);
  return hash;
}

static char*
copy_path(char** cursor, const char* path)
{
  size_t length = strlen(path) + 1;
  char* result = *cursor;
  memcpy(result, path, length);
  *cursor += length;
  return result;
}

enum damacy_status
prepared_plan_build(struct prefetch_cache* arrays,
                    struct prefetch_cache* shards,
                    const struct planner_sample* samples,
                    uint32_t n_samples,
                    const struct damacy_batch_spec* output,
                    const struct damacy_plan_limits* limits,
                    struct prepared_plan** out)
{
  if (!out)
    return DAMACY_INVAL;
  *out = NULL;
  if (!arrays || !shards || !samples || !output || !limits || !n_samples ||
      !limits->max_chunks || !limits->max_chunk_bytes ||
      output->sample_rank == 0 || output->sample_rank > DAMACY_MAX_RANK)
    return DAMACY_INVAL;
  enum damacy_status status = DAMACY_OK;
  struct prepared_plan* plan = NULL;
  struct strbuf path = { 0 };
  uint64_t capacity = 0;
  uint64_t path_bytes = 0;
  uint64_t index_bytes = 0;
  for (uint32_t i = 0; i < n_samples; ++i) {
    const struct zarr_metadata* meta = NULL;
    if (!samples[i].uri || samples[i].n_shards > limits->max_shards_per_sample)
      return DAMACY_INVAL;
    status = array_metadata(arrays, samples[i].h_meta, &meta);
    if (status != DAMACY_OK)
      return status;
    uint32_t count, bytes;
    status = sample_geometry(&samples[i], meta, output, limits, &count, &bytes);
    if (status != DAMACY_OK)
      return status;
    for (uint8_t d = 0; d < meta->rank; ++d) {
      uint64_t size =
        (uint64_t)samples[i].axes[d].count * sizeof(struct query_index);
      if (size > limits->max_plan_bytes ||
          index_bytes > limits->max_plan_bytes - size)
        return DAMACY_BUDGET;
      index_bytes += size;
    }
    capacity += count;
    if (capacity > limits->max_chunks)
      return DAMACY_BUDGET;
    uint64_t length = strlen(samples[i].uri) + 1;
    uint64_t per_path = length + 3 + (uint64_t)meta->rank * 21;
    if (per_path > limits->max_plan_bytes ||
        samples[i].n_shards > limits->max_plan_bytes / per_path)
      return DAMACY_BUDGET;
    uint64_t addition = length + per_path * samples[i].n_shards;
    if (addition > limits->max_plan_bytes ||
        path_bytes > limits->max_plan_bytes - addition)
      return DAMACY_BUDGET;
    path_bytes += addition;
  }
  uint64_t buckets = 1;
  while (buckets < 2 * capacity)
    buckets *= 2;
  uint64_t storage_bytes =
    (uint64_t)n_samples *
      (sizeof(struct plan_array) + sizeof(struct plan_region)) +
    capacity * (sizeof(struct plan_chunk) + sizeof(struct plan_use)) +
    buckets * sizeof(uint32_t) + path_bytes;
  if (index_bytes > UINT64_MAX - storage_bytes)
    return DAMACY_BUDGET;
  storage_bytes += index_bytes;
  if (storage_bytes > SIZE_MAX || storage_bytes > limits->max_plan_bytes ||
      sizeof(*plan) > limits->max_plan_bytes - storage_bytes)
    return DAMACY_BUDGET;
  plan = calloc(1, sizeof(*plan));
  if (!plan)
    return DAMACY_OOM;
  plan->storage = calloc(1, (size_t)storage_bytes);
  if (!plan->storage) {
    status = DAMACY_OOM;
    goto Done;
  }
  plan->allocated_bytes = sizeof(*plan) + storage_bytes;
  plan->output = *output;
  plan->arrays = plan->storage;
  plan->regions = (void*)(plan->arrays + n_samples);
  plan->chunks = (void*)(plan->regions + n_samples);
  struct query_index* indices = (void*)(plan->chunks + capacity);
  plan->uses = (void*)((char*)indices + index_bytes);
  uint32_t* table = (void*)(plan->uses + capacity);
  char* paths = (void*)(table + buckets);
  for (uint32_t i = 0; i < n_samples; ++i) {
    const struct planner_sample* sample = &samples[i];
    const struct zarr_metadata* meta = NULL;
    status = array_metadata(arrays, sample->h_meta, &meta);
    if (status != DAMACY_OK)
      goto Done;
    uint32_t array = 0;
    while (array < plan->n_arrays &&
           strcmp(plan->arrays[array].uri, sample->uri))
      ++array;
    if (array == plan->n_arrays) {
      plan->arrays[array].uri = copy_path(&paths, sample->uri);
      plan->arrays[array].metadata = *meta;
      ++plan->n_arrays;
    }
    plan->regions[i] = (struct plan_region){ .operation = PLAN_COPY,
                                             .array = array,
                                             .sample = i,
                                             .source = sample->aabb };
    for (uint8_t d = 0; d < meta->rank; ++d) {
      uint32_t n = sample->axes[d].count;
      if (!n)
        continue;
      memcpy(indices, sample->axes[d].indices, (size_t)n * sizeof(*indices));
      plan->regions[i].axes[d] = (struct query_axis){ indices, n };
      plan->regions[i].operation = PLAN_GATHER;
      indices += n;
    }
    ++plan->n_regions;
    uint64_t per_shard[DAMACY_MAX_RANK];
    uint32_t count, decoded_bytes;
    status =
      sample_geometry(sample, meta, output, limits, &count, &decoded_bytes);
    if (status != DAMACY_OK)
      goto Done;
    if (zarr_metadata_inner_per_shard(meta, per_shard, NULL)) {
      status = DAMACY_DECODE;
      goto Done;
    }
    struct selection_grid iterator;
    status = selection_grid_init(&iterator,
                                 &sample->aabb,
                                 sample->axes,
                                 meta->shard_shape,
                                 NULL,
                                 NULL,
                                 limits->max_shards_per_sample);
    if (status != DAMACY_OK)
      goto Done;
    uint64_t shard_coord[DAMACY_MAX_RANK];
    uint32_t shard_index = 0;
    while (selection_grid_next(&iterator, shard_coord)) {
      if (shard_index >= sample->n_shards) {
        status = DAMACY_INVAL;
        goto Done;
      }
      const void* value = NULL;
      int error = 0;
      enum prefetch_state state = prefetch_cache_query(
        shards, sample->h_shards[shard_index++], &value, &error);
      int missing = state == PREFETCH_STATE_ERROR && error == DAMACY_NOTFOUND;
      if (!missing && (state != PREFETCH_STATE_READY || !value)) {
        status = state == PREFETCH_STATE_PENDING ? DAMACY_AGAIN
                 : error                         ? (enum damacy_status)error
                                                 : DAMACY_DECODE;
        goto Done;
      }
      const struct shard_index_value* index = value;
      const char* shard_path = NULL;
      if (!missing) {
        if (zarr_shard_path_build(
              &path, sample->uri, shard_coord, meta->rank)) {
          status = DAMACY_OOM;
          goto Done;
        }
        shard_path = copy_path(&paths, strbuf_cstr(&path));
      }
      uint64_t lo[DAMACY_MAX_RANK], hi[DAMACY_MAX_RANK];
      uint64_t coordinate[DAMACY_MAX_RANK] = { 0 };
      for (uint8_t d = 0; d < meta->rank; ++d) {
        lo[d] = shard_coord[d] * per_shard[d];
        hi[d] = lo[d] + per_shard[d];
      }
      struct selection_grid chunks;
      status = selection_grid_init(&chunks,
                                   &sample->aabb,
                                   sample->axes,
                                   meta->inner_chunk_shape,
                                   lo,
                                   hi,
                                   limits->max_chunks);
      if (status != DAMACY_OK)
        goto Done;
      while (selection_grid_next(&chunks, coordinate)) {
        uint64_t entry_index = 0;
        for (uint8_t d = 0; d < meta->rank; ++d)
          entry_index =
            entry_index * per_shard[d] + coordinate[d] % per_shard[d];
        struct plan_chunk chunk = { .array = array,
                                    .path = shard_path,
                                    .decoded_bytes = decoded_bytes,
                                    .first_use = UINT32_MAX,
                                    .missing = (uint8_t)missing };
        memcpy(chunk.coordinate, coordinate, sizeof(coordinate));
        if (!missing) {
          if (entry_index >= index->n_entries) {
            status = DAMACY_DECODE;
            goto Done;
          }
          const struct zarr_shard_entry* entry = &index->entries[entry_index];
          int empty_offset = entry->offset == ZARR_SHARD_EMPTY_OFFSET;
          int empty_size = entry->nbytes == ZARR_SHARD_EMPTY_NBYTES;
          if (empty_offset != empty_size ||
              (!empty_size && (!entry->nbytes || entry->nbytes > UINT32_MAX ||
                               entry->offset > UINT64_MAX - entry->nbytes))) {
            status = DAMACY_DECODE;
            goto Done;
          }
          chunk.missing = (uint8_t)empty_size;
          if (!empty_size) {
            chunk.offset = entry->offset;
            chunk.encoded_bytes = (uint32_t)entry->nbytes;
          }
        }
        uint64_t bucket =
          chunk_hash(array, coordinate, meta->rank) & (buckets - 1);
        while (table[bucket]) {
          const struct plan_chunk* found = &plan->chunks[table[bucket] - 1];
          if (found->array == array &&
              !memcmp(found->coordinate, coordinate, sizeof(coordinate)))
            break;
          bucket = (bucket + 1) & (buckets - 1);
        }
        uint32_t chunk_index;
        if (!table[bucket]) {
          chunk_index = plan->n_chunks++;
          plan->chunks[chunk_index] = chunk;
          table[bucket] = chunk_index + 1;
        } else {
          chunk_index = table[bucket] - 1;
        }
        if (plan->n_uses >= capacity) {
          status = DAMACY_BUDGET;
          goto Done;
        }
        struct plan_chunk* stored = &plan->chunks[chunk_index];
        plan->uses[plan->n_uses] = (struct plan_use){
          .chunk = chunk_index, .region = i, .next = stored->first_use
        };
        stored->first_use = plan->n_uses++;
      }
    }
    if (shard_index != sample->n_shards) {
      status = DAMACY_INVAL;
      goto Done;
    }
  }
Done:
  strbuf_free(&path);
  if (status != DAMACY_OK)
    prepared_plan_destroy(plan);
  else
    *out = plan;
  return status;
}
