#include "damacy_stats.h"
#include "expect.h"
#include "pipeline/components.h"
#include "store/store_internal.h"

#include <stdlib.h>
#include <string.h>

enum
{
  MAX_CHUNKS = 32,
  MAX_READS = 64,
  SHARD_BYTES = 128,
  CHUNK_BYTES = 8,
};

struct read_record
{
  unsigned shard;
  uint64_t offset;
  size_t bytes;
};

struct pending_reads
{
  struct store_read reads[MAX_CHUNKS];
  size_t count;
  int active;
};

struct test_store
{
  struct store base;
  uint16_t data[3][SHARD_BYTES / sizeof(uint16_t)];
  struct read_record records[MAX_READS];
  struct pending_reads events[MAX_READS];
  unsigned n_reads;
  unsigned n_events;
  unsigned pending;
  unsigned capacity;
  unsigned rejected;
  unsigned waited;
  unsigned completion_errors;
  uint64_t held_event;
  enum damacy_status submit_status;
  enum damacy_status read_status;
};

static enum damacy_status
complete_reads(struct test_store* store, struct store_event event)
{
  if (!event.impl)
    return DAMACY_OK;
  struct pending_reads* pending = event.impl;
  enum damacy_status status = store->read_status;
  for (size_t i = 0; i < pending->count && status == DAMACY_OK; ++i) {
    const struct store_read* read = &pending->reads[i];
    unsigned shard = (unsigned)(read->key[0] - 'a');
    if (shard >= 3 || read->offset > SHARD_BYTES ||
        read->len > SHARD_BYTES - read->offset) {
      status = DAMACY_IO;
      break;
    }
    memcpy(read->dst, (char*)store->data[shard] + read->offset, read->len);
  }
  store->pending -= (unsigned)pending->count;
  pending->active = 0;
  if (status != DAMACY_OK)
    ++store->completion_errors;
  return status;
}

static struct store_submit_result
submit_reads(struct store* base, const struct store_read* reads, size_t count)
{
  struct test_store* store = (void*)base;
  if (store->submit_status != DAMACY_OK)
    return (struct store_submit_result){ .status = store->submit_status };
  if (!count)
    return (struct store_submit_result){ .status = DAMACY_OK };
  if (count > store->capacity - store->pending) {
    ++store->rejected;
    return (struct store_submit_result){ .status = DAMACY_AGAIN };
  }
  if (count > MAX_CHUNKS || count > MAX_READS - store->n_reads ||
      store->n_events == MAX_READS)
    return (struct store_submit_result){ .status = DAMACY_IO };
  struct pending_reads* pending = &store->events[store->n_events++];
  pending->count = count;
  pending->active = 1;
  memcpy(pending->reads, reads, count * sizeof(*reads));
  for (size_t i = 0; i < count; ++i)
    store->records[store->n_reads++] =
      (struct read_record){ .shard = (unsigned)(reads[i].key[0] - 'a'),
                            .offset = reads[i].offset,
                            .bytes = reads[i].len };
  store->pending += (unsigned)count;
  return (struct store_submit_result){
    .status = DAMACY_OK, .event = { .seq = store->n_events, .impl = pending }
  };
}

static struct store_event_poll
query_reads(struct store* base, struct store_event event)
{
  struct test_store* store = (void*)base;
  if (event.impl && event.seq == store->held_event)
    return (struct store_event_poll){ .status = DAMACY_OK };
  return (struct store_event_poll){ .status = complete_reads(store, event),
                                    .ready = 1 };
}

static enum damacy_status
wait_reads(struct store* base, struct store_event event)
{
  struct test_store* store = (void*)base;
  ++store->waited;
  return complete_reads(store, event);
}

static const struct store_vtable test_store_ops = { .submit = submit_reads,
                                                    .event_query = query_reads,
                                                    .event_wait = wait_reads };

static void
store_init(struct test_store* store, unsigned capacity)
{
  *store = (struct test_store){ .base = { .vt = &test_store_ops },
                                .capacity = capacity };
  for (unsigned shard = 0; shard < 3; ++shard)
    for (unsigned i = 0; i < SHARD_BYTES / sizeof(uint16_t); ++i)
      store->data[shard][i] = (uint16_t)(100 * shard + i);
}

struct test_chunk
{
  unsigned shard;
  uint64_t offset;
  uint8_t missing;
};

struct plan_storage
{
  struct plan_array array;
  struct plan_chunk chunks[MAX_CHUNKS];
  struct plan_region regions[2];
  struct plan_use uses[2 * MAX_CHUNKS];
  char paths[MAX_CHUNKS][2];
};

static struct damacy_batch_spec
output_spec(uint32_t count)
{
  return (struct damacy_batch_spec){ .dtype = DAMACY_F32,
                                     .sample_shape = { count * 4 },
                                     .sample_rank = 1,
                                     .samples_per_batch = 2 };
}

static struct prepared_plan*
make_plan(const struct test_chunk* chunks, uint32_t count)
{
  if (!count || count > MAX_CHUNKS)
    return NULL;
  struct prepared_plan* plan = calloc(1, sizeof(*plan));
  struct plan_storage* storage = calloc(1, sizeof(*storage));
  if (!plan || !storage) {
    free(plan);
    free(storage);
    return NULL;
  }
  *plan =
    (struct prepared_plan){ .output = output_spec(count),
                            .arrays = &storage->array,
                            .chunks = storage->chunks,
                            .regions = storage->regions,
                            .uses = storage->uses,
                            .n_arrays = 1,
                            .n_chunks = count,
                            .n_regions = 2,
                            .n_uses = 2 * count,
                            .allocated_bytes = sizeof(*plan) + sizeof(*storage),
                            .storage = storage };
  storage->array.metadata =
    (struct zarr_metadata){ .rank = 1,
                            .dtype = dtype_u16,
                            .shape = { count * 4 },
                            .inner_chunk_shape = { 4 },
                            .inner_codec = { .id = CODEC_NONE } };
  uint16_t fill = 999;
  memcpy(storage->array.metadata.fill_value, &fill, sizeof(fill));
  for (unsigned sample = 0; sample < 2; ++sample)
    storage->regions[sample] = (struct plan_region){
      .sample = sample, .source = { .rank = 1, .dims = { { 0, count * 4 } } }
    };
  for (uint32_t i = 0; i < count; ++i) {
    storage->paths[i][0] = (char)('a' + chunks[i].shard);
    storage->chunks[i] =
      (struct plan_chunk){ .path = storage->paths[i],
                           .offset = chunks[i].offset,
                           .coordinate = { i },
                           .encoded_bytes = chunks[i].missing ? 0 : CHUNK_BYTES,
                           .decoded_bytes = CHUNK_BYTES,
                           .first_use = 2 * i,
                           .missing = chunks[i].missing };
    storage->uses[2 * i] =
      (struct plan_use){ .chunk = i, .region = 0, .next = 2 * i + 1 };
    storage->uses[2 * i + 1] =
      (struct plan_use){ .chunk = i, .region = 1, .next = UINT32_MAX };
  }
  return plan;
}

static int
start_executor(struct damacy_reader* reader,
               uint32_t workers,
               uint32_t count,
               uint64_t budget,
               struct damacy_stats* stats,
               struct damacy_executor** out)
{
  struct damacy_cpu_config config = { .decode_workers = workers,
                                      .max_encoded_chunk_bytes = CHUNK_BYTES,
                                      .max_decoded_chunk_bytes = CHUNK_BYTES,
                                      .max_memory_bytes = budget };
  EXPECT(damacy_cpu_executor_create(reader, &config, out) == DAMACY_OK);
  struct damacy_batch_spec output = output_spec(count);
  EXPECT((*out)->ops->start(*out, &output, stats) == DAMACY_OK);
  return 0;
}

static int
test_read_errors_and_shutdown(void)
{
  const struct test_chunk chunks[] = {
    { .offset = 0 }, { .offset = 8 }, { .offset = 16 }, { .offset = 24 }
  };
  for (unsigned failure = 0; failure < 3; ++failure) {
    struct test_store store;
    store_init(&store, 4);
    if (failure == 0)
      store.submit_status = DAMACY_IO;
    else if (failure == 1)
      store.read_status = DAMACY_IO;
    else
      store.held_event = 2;
    struct damacy_reader reader = { .store = &store.base,
                                    .max_inflight_reads = 2 };
    struct damacy_stats stats = { 0 };
    struct damacy_executor* executor = NULL;
    EXPECT(start_executor(&reader, 2, 4, 8 << 20, &stats, &executor) == 0);
    struct prepared_plan* plan = make_plan(chunks, 4);
    EXPECT(plan);
    EXPECT(executor->ops->submit(executor, plan, 0) == DAMACY_OK);
    int changed = 0;
    EXPECT(executor->ops->step(executor, &changed) ==
           (failure == 2 ? DAMACY_OK : DAMACY_IO));
    struct damacy_batch* batch = NULL;
    EXPECT(executor->ops->take(executor, &batch) == DAMACY_AGAIN);
    damacy_executor_destroy(executor);
    EXPECT(store.pending == 0);
    if (failure)
      EXPECT(store.waited == 1);
    if (failure == 2)
      EXPECT(store.completion_errors == 0);
  }
  return 0;
}

int
main(void)
{
  RUN(test_read_errors_and_shutdown);
  return 0;
}
