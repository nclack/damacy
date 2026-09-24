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
finish_batch(struct damacy_executor* executor, struct damacy_batch** out)
{
  for (unsigned i = 0; i < 2 * MAX_CHUNKS; ++i) {
    int changed = 0;
    EXPECT(executor->ops->step(executor, &changed) == DAMACY_OK);
    enum damacy_status status = executor->ops->take(executor, out);
    if (status == DAMACY_OK)
      return 0;
    EXPECT(status == DAMACY_AGAIN && changed);
  }
  return 1;
}

static int
check_output(struct damacy_batch* batch,
             const struct test_store* store,
             const struct test_chunk* chunks,
             uint32_t count)
{
  struct damacy_batch_info info;
  damacy_batch_info(batch, &info);
  EXPECT(info.device_type == DAMACY_DEVICE_CPU);
  EXPECT(info.rank == 2 && info.shape[0] == 2 && info.shape[1] == count * 4);
  const float* data = info.data;
  for (unsigned sample = 0; sample < 2; ++sample)
    for (uint32_t c = 0; c < count; ++c)
      for (unsigned i = 0; i < 4; ++i) {
        float expected =
          chunks[c].missing
            ? 999
            : store->data[chunks[c].shard][chunks[c].offset / 2 + i];
        EXPECT(data[(sample * count + c) * 4 + i] == expected);
      }
  return 0;
}

static int
test_merge_and_shard_order(void)
{
  struct test_chunk chunks[12];
  for (unsigned shard = 0; shard < 3; ++shard)
    for (unsigned i = 0; i < 4; ++i)
      chunks[shard * 4 + i] =
        (struct test_chunk){ .shard = shard,
                             .offset = (i < 2 ? 40 : 8) - (i % 2) * 8 };
  struct test_store store;
  store_init(&store, 4);
  struct damacy_reader reader = { .store = &store.base,
                                  .max_inflight_reads = 4 };
  struct damacy_stats stats = { 0 };
  struct damacy_executor* executor = NULL;
  EXPECT(start_executor(&reader, 4, 12, 8 << 20, &stats, &executor) == 0);
  struct prepared_plan* plan = make_plan(chunks, 12);
  EXPECT(plan);
  EXPECT(plan->chunks[0].path != plan->chunks[1].path);
  EXPECT(executor->ops->submit(executor, plan, 0) == DAMACY_OK);
  struct damacy_batch* batch = NULL;
  EXPECT(finish_batch(executor, &batch) == 0);
  EXPECT(check_output(batch, &store, chunks, 12) == 0);
  EXPECT(store.n_reads == 6);
  for (unsigned i = 0; i < store.n_reads; ++i) {
    EXPECT(store.records[i].shard == i % 3);
    EXPECT(store.records[i].offset == (i < 3 ? 0 : 32));
    EXPECT(store.records[i].bytes == 2 * CHUNK_BYTES);
  }
  EXPECT(stats.reads_issued == 6 && stats.chunks_dispatched == 12);
  EXPECT(stats.decode.count == 12);
  EXPECT(stats.decode.input_bytes == 12 * CHUNK_BYTES);
  EXPECT(stats.decode.output_bytes == 12 * CHUNK_BYTES);
  EXPECT(stats.io.input_bytes == 12 * CHUNK_BYTES);
  damacy_batch_release(batch);
  damacy_executor_destroy(executor);
  return 0;
}

static int
test_reader_limit_and_retry(void)
{
  const struct test_chunk chunks[] = {
    { .offset = 0 }, { .offset = 8 }, { .offset = 16 }, { .offset = 24 }
  };
  struct test_store store;
  store_init(&store, 1);
  struct damacy_reader reader = { .store = &store.base,
                                  .max_inflight_reads = 1 };
  struct damacy_stats stats = { 0 };
  struct damacy_executor* executor = NULL;
  EXPECT(start_executor(&reader, 2, 4, 8 << 20, &stats, &executor) == 0);
  struct prepared_plan* plan = make_plan(chunks, 4);
  EXPECT(plan);
  EXPECT(executor->ops->submit(executor, plan, 0) == DAMACY_OK);
  struct damacy_batch* batch = NULL;
  EXPECT(finish_batch(executor, &batch) == 0);
  EXPECT(check_output(batch, &store, chunks, 4) == 0);
  EXPECT(store.rejected > 0 && store.pending == 0 && store.n_reads == 2);
  EXPECT(store.records[0].offset == 0 && store.records[0].bytes == 16);
  EXPECT(store.records[1].offset == 16 && store.records[1].bytes == 16);
  EXPECT(stats.chunks_dispatched == 4 && stats.reads_issued == 2);
  damacy_batch_release(batch);
  damacy_executor_destroy(executor);
  return 0;
}

static int
test_overlapping_ranges(void)
{
  const struct test_chunk chunks[] = { { .offset = 4 }, { .offset = 0 } };
  struct test_store store;
  store_init(&store, 2);
  struct damacy_reader reader = { .store = &store.base,
                                  .max_inflight_reads = 2 };
  struct damacy_stats stats = { 0 };
  struct damacy_executor* executor = NULL;
  EXPECT(start_executor(&reader, 2, 2, 8 << 20, &stats, &executor) == 0);
  struct prepared_plan* plan = make_plan(chunks, 2);
  EXPECT(plan);
  EXPECT(executor->ops->submit(executor, plan, 0) == DAMACY_OK);
  struct damacy_batch* batch = NULL;
  EXPECT(finish_batch(executor, &batch) == 0);
  EXPECT(check_output(batch, &store, chunks, 2) == 0);
  EXPECT(store.n_reads == 1 && store.records[0].offset == 0);
  EXPECT(store.records[0].bytes == 12);
  EXPECT(stats.decode.input_bytes == 16 && stats.io.input_bytes == 12);
  damacy_batch_release(batch);
  damacy_executor_destroy(executor);
  return 0;
}

static int
test_single_worker_and_fills(void)
{
  for (unsigned all_missing = 0; all_missing < 2; ++all_missing) {
    const struct test_chunk chunks[] = {
      { .missing = 1 },
      { .shard = 2, .offset = 24, .missing = (uint8_t)all_missing },
      { .missing = 1 },
      { .shard = 0, .offset = 8, .missing = (uint8_t)all_missing },
    };
    struct test_store store;
    store_init(&store, 1);
    struct damacy_reader reader = { .store = &store.base,
                                    .max_inflight_reads = 1 };
    struct damacy_stats stats = { 0 };
    struct damacy_executor* executor = NULL;
    EXPECT(start_executor(&reader, 1, 4, 8 << 20, &stats, &executor) == 0);
    struct prepared_plan* plan = make_plan(chunks, 4);
    EXPECT(plan);
    EXPECT(executor->ops->submit(executor, plan, 0) == DAMACY_OK);
    struct damacy_batch* batch = NULL;
    EXPECT(finish_batch(executor, &batch) == 0);
    EXPECT(check_output(batch, &store, chunks, 4) == 0);
    EXPECT(store.n_reads == (all_missing ? 0u : 2u));
    EXPECT(stats.chunks_dispatched == 4);
    damacy_batch_release(batch);
    damacy_executor_destroy(executor);
  }
  return 0;
}

static int
test_fills_share_merged_input(void)
{
  const struct test_chunk chunks[] = {
    { .shard = 1, .offset = 16 }, { .missing = 1 },
    { .shard = 1, .offset = 24 }, { .missing = 1 },
    { .shard = 0, .offset = 0 },  { .shard = 0, .offset = 8 },
  };
  struct test_store store;
  store_init(&store, 2);
  struct damacy_reader reader = { .store = &store.base,
                                  .max_inflight_reads = 2 };
  struct damacy_stats stats = { 0 };
  struct damacy_executor* executor = NULL;
  EXPECT(start_executor(&reader, 6, 6, 8 << 20, &stats, &executor) == 0);
  struct prepared_plan* plan = make_plan(chunks, 6);
  EXPECT(plan);
  EXPECT(executor->ops->submit(executor, plan, 0) == DAMACY_OK);
  struct damacy_batch* batch = NULL;
  EXPECT(finish_batch(executor, &batch) == 0);
  EXPECT(check_output(batch, &store, chunks, 6) == 0);
  EXPECT(store.n_events == 1 && store.events[0].count == 2);
  EXPECT(store.records[0].shard == 0 && store.records[0].offset == 0);
  EXPECT(store.records[1].shard == 1 && store.records[1].offset == 16);
  EXPECT(store.records[0].bytes == 2 * CHUNK_BYTES);
  EXPECT(store.records[1].bytes == 2 * CHUNK_BYTES);
  EXPECT(stats.waves_emitted == 1 && stats.reads_issued == 2);
  EXPECT(stats.chunks_dispatched == 6 && stats.decode.count == 6);
  EXPECT(stats.decode.input_bytes == 4 * CHUNK_BYTES);
  EXPECT(stats.io.input_bytes == 4 * CHUNK_BYTES);
  damacy_batch_release(batch);
  damacy_executor_destroy(executor);
  return 0;
}

static int
test_batch_order_and_retained_output(void)
{
  const struct test_chunk chunks[] = { { .offset = 8 }, { .offset = 0 } };
  struct test_store store;
  store_init(&store, 2);
  store.held_event = 1;
  struct damacy_reader reader = { .store = &store.base,
                                  .max_inflight_reads = 2 };
  struct damacy_stats stats = { 0 };
  struct damacy_executor* executor = NULL;
  EXPECT(start_executor(&reader, 2, 2, 8 << 20, &stats, &executor) == 0);
  for (unsigned i = 0; i < 2; ++i) {
    struct prepared_plan* plan = make_plan(chunks, 2);
    EXPECT(plan);
    EXPECT(executor->ops->submit(executor, plan, i) == DAMACY_OK);
  }
  int changed = 0;
  EXPECT(executor->ops->step(executor, &changed) == DAMACY_OK);
  EXPECT(store.n_events == 2 && store.pending == 1);
  struct damacy_batch* first = NULL;
  EXPECT(executor->ops->take(executor, &first) == DAMACY_AGAIN);
  store.held_event = 0;
  EXPECT(finish_batch(executor, &first) == 0);
  struct damacy_batch* second = NULL;
  EXPECT(executor->ops->take(executor, &second) == DAMACY_OK);
  struct damacy_batch_info a, b;
  damacy_batch_info(first, &a);
  damacy_batch_info(second, &b);
  EXPECT(a.batch_id == 0 && b.batch_id == 1 && a.data != b.data);
  struct prepared_plan* third = make_plan(chunks, 2);
  EXPECT(third);
  EXPECT(executor->ops->submit(executor, third, 2) == DAMACY_AGAIN);
  damacy_batch_retain(first);
  damacy_batch_release(first);
  damacy_batch_release(second);
  EXPECT(executor->ops->step(executor, &changed) == DAMACY_OK);
  EXPECT(executor->ops->submit(executor, third, 2) == DAMACY_OK);
  struct damacy_batch* last = NULL;
  EXPECT(finish_batch(executor, &last) == 0);
  EXPECT(check_output(last, &store, chunks, 2) == 0);
  damacy_batch_release(last);
  damacy_executor_destroy(executor);
  EXPECT(check_output(first, &store, chunks, 2) == 0);
  damacy_batch_release(first);
  return 0;
}

static int
test_plan_memory_budget(void)
{
  const struct test_chunk chunks[] = { { .offset = 0 }, { .offset = 8 } };
  struct test_store store;
  store_init(&store, 2);
  struct damacy_reader reader = { .store = &store.base,
                                  .max_inflight_reads = 2 };
  struct damacy_stats stats = { 0 };
  struct damacy_executor* executor = NULL;
  EXPECT(start_executor(&reader, 2, 2, 8 << 20, &stats, &executor) == 0);
  executor->ops->stats(executor, &stats);
  uint64_t initial = stats.host_bytes_committed;
  struct prepared_plan* plan = make_plan(chunks, 2);
  EXPECT(plan);
  EXPECT(executor->ops->submit(executor, plan, 0) == DAMACY_OK);
  executor->ops->stats(executor, &stats);
  uint64_t with_plan = stats.host_bytes_committed;
  EXPECT(with_plan > initial && with_plan <= (8 << 20));
  struct damacy_batch* batch = NULL;
  EXPECT(finish_batch(executor, &batch) == 0);
  executor->ops->stats(executor, &stats);
  EXPECT(stats.host_bytes_committed == initial);
  damacy_batch_release(batch);
  damacy_executor_destroy(executor);
  executor = NULL;
  EXPECT(start_executor(&reader, 2, 2, with_plan, &stats, &executor) == 0);
  plan = make_plan(chunks, 2);
  EXPECT(plan);
  EXPECT(executor->ops->submit(executor, plan, 0) == DAMACY_BUDGET);
  executor->ops->stats(executor, &stats);
  EXPECT(stats.host_bytes_committed == initial);
  prepared_plan_destroy(plan);
  damacy_executor_destroy(executor);
  return 0;
}

static int
test_plan_memory_retry(void)
{
  const struct test_chunk chunks[] = { { .offset = 0 }, { .offset = 8 } };
  struct test_store store;
  store_init(&store, 2);
  struct damacy_reader reader = { .store = &store.base,
                                  .max_inflight_reads = 2 };
  struct damacy_stats stats = { 0 };
  struct damacy_executor* executor = NULL;
  EXPECT(start_executor(&reader, 2, 2, 8 << 20, &stats, &executor) == 0);
  executor->ops->stats(executor, &stats);
  uint64_t initial = stats.host_bytes_committed;
  struct prepared_plan* plan = make_plan(chunks, 2);
  EXPECT(plan);
  EXPECT(executor->ops->submit(executor, plan, 0) == DAMACY_OK);
  executor->ops->stats(executor, &stats);
  uint64_t plan_bytes = stats.host_bytes_committed - initial;
  EXPECT(plan_bytes > 0);
  damacy_executor_destroy(executor);
  for (unsigned i = 1; i <= 16; ++i) {
    executor = NULL;
    EXPECT(start_executor(
             &reader, 2, 2, initial + i * plan_bytes, &stats, &executor) == 0);
    plan = make_plan(chunks, 2);
    EXPECT(plan);
    enum damacy_status status = executor->ops->submit(executor, plan, 0);
    if (status == DAMACY_BUDGET) {
      prepared_plan_destroy(plan);
      damacy_executor_destroy(executor);
      continue;
    }
    EXPECT(status == DAMACY_OK);
    struct prepared_plan* next = make_plan(chunks, 2);
    EXPECT(next);
    EXPECT(executor->ops->submit(executor, next, 1) == DAMACY_AGAIN);
    struct damacy_batch* first = NULL;
    EXPECT(finish_batch(executor, &first) == 0);
    EXPECT(executor->ops->submit(executor, next, 1) == DAMACY_OK);
    struct damacy_batch* second = NULL;
    EXPECT(finish_batch(executor, &second) == 0);
    EXPECT(check_output(first, &store, chunks, 2) == 0);
    EXPECT(check_output(second, &store, chunks, 2) == 0);
    damacy_batch_release(first);
    damacy_batch_release(second);
    damacy_executor_destroy(executor);
    return 0;
  }
  return 1;
}

static int
test_read_errors_and_shutdown(void)
{
  const struct test_chunk chunks[] = {
    { .offset = 0 }, { .offset = 8 }, { .offset = 16 }, { .offset = 24 }
  };
  for (unsigned failure = 0; failure < 3; ++failure) {
    struct test_store store;
    store_init(&store, 2);
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
  RUN(test_merge_and_shard_order);
  RUN(test_reader_limit_and_retry);
  RUN(test_overlapping_ranges);
  RUN(test_single_worker_and_fills);
  RUN(test_fills_share_merged_input);
  RUN(test_batch_order_and_retained_output);
  RUN(test_plan_memory_budget);
  RUN(test_plan_memory_retry);
  RUN(test_read_errors_and_shutdown);
  return 0;
}
