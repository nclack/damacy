#include "damacy_pipeline.h"
#include "fixture.h"
#include "pipeline/components.h"
#include "platform/platform.h"

#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

struct components
{
  struct damacy_reader* reader;
  struct damacy_metadata_reader* metadata_reader;
  struct damacy_metadata* metadata;
  struct damacy_planner* planner;
  struct damacy_executor* executor;
  struct damacy* pipeline;
};

static void
destroy_components(struct components* c)
{
  damacy_destroy(c->pipeline);
  damacy_executor_destroy(c->executor);
  damacy_planner_destroy(c->planner);
  damacy_metadata_destroy(c->metadata);
  damacy_metadata_reader_destroy(c->metadata_reader);
  damacy_reader_destroy(c->reader);
  *c = (struct components){ 0 };
}

static int
create_components(struct components* c)
{
  EXPECT(damacy_file_reader_create(2, 4, &c->reader) == DAMACY_OK);
  EXPECT(damacy_file_metadata_reader_create(4, NULL, &c->metadata_reader) ==
         DAMACY_OK);
  EXPECT(
    damacy_zarr_metadata_create(c->metadata_reader,
                                &(struct damacy_metadata_cache_config){
                                  .array_entries = 8, .shard_entries = 32 },
                                &c->metadata) == DAMACY_OK);
  EXPECT(damacy_chunk_planner_create(
           c->metadata,
           &(struct damacy_plan_limits){ .max_chunks = 1024,
                                         .max_chunk_bytes = 1 << 20,
                                         .max_shards_per_sample = 4,
                                         .max_plan_bytes = 1 << 20 },
           &c->planner) == DAMACY_OK);
  EXPECT(damacy_cpu_executor_create(c->reader,
                                    &(struct damacy_cpu_config){
                                      .decode_workers = 2,
                                      .max_encoded_chunk_bytes = (1 << 20) + 1,
                                      .max_decoded_chunk_bytes = 1 << 20,
                                      .max_memory_bytes = 32 << 20 },
                                    &c->executor) == DAMACY_OK);
  return 0;
}

static int
start_pipeline(struct components* c,
               enum damacy_dtype dtype,
               int rows,
               int cols,
               uint32_t samples)
{
  EXPECT(damacy_pipeline_create(
           c->planner,
           c->executor,
           &(struct damacy_batch_spec){ .dtype = dtype,
                                        .sample_shape = { rows, cols },
                                        .sample_rank = 2,
                                        .samples_per_batch = samples },
           &(struct damacy_queue_limits){ .lookahead_samples = 4,
                                          .prepared_batches = 2 },
           &c->pipeline) == DAMACY_OK);
  return 0;
}

static struct damacy_sample
sample(const char* uri, int y, int x, int rows, int cols)
{
  return (struct damacy_sample){
    .uri = uri,
    .aabb = { .rank = 2, .dims = { { y, y + rows }, { x, x + cols } } }
  };
}

static int
verify_crop(struct damacy_batch* batch, int offset, int unsigned_bits)
{
  struct damacy_batch_info info;
  damacy_batch_info(batch, &info);
  EXPECT(info.device_type == DAMACY_DEVICE_CPU);
  EXPECT(info.device_id == 0);
  EXPECT(info.data == info.device_ptr);
  EXPECT(info.ready_stream == NULL);
  EXPECT(info.rank == 3 && info.shape[0] == 2 && info.shape[1] == 2 &&
         info.shape[2] == 5);
  const float* values = info.data;
  for (unsigned s = 0; s < 2; ++s)
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 5; ++x) {
        int64_t expected = (y + 1) * 11 + x + 2 + offset;
        if (unsigned_bits)
          expected &= ((1ll << unsigned_bits) - 1);
        EXPECT(values[s * 10 + y * 5 + x] == (float)expected);
      }
  return 0;
}

static int
test_codecs_and_types(void)
{
  const char* codecs[] = { "none", "zstd", "blosc-zstd" };
  const char* types[] = { "uint8", "uint16",  "int16",  "uint32",
                          "int32", "float16", "float32" };
  const int unsigned_bits[] = { 8, 16, 0, 32, 0, 0, 0 };
  for (unsigned ci = 0; ci < 3; ++ci)
    for (unsigned ti = 0; ti < 7; ++ti) {
      char root[] = "/tmp/damacy_cpu_XXXXXX";
      EXPECT(mkdtemp(root));
      char uri[256];
      snprintf(uri, sizeof(uri), "%s/array", root);
      int64_t shape[] = { 5, 11 }, chunks[] = { 2, 4 }, shards[] = { 4, 8 };
      EXPECT(fixture_write_zarr_codec(
               uri, shape, chunks, shards, 2, types[ti], -20, codecs[ci]) == 0);
      struct components c = { 0 };
      EXPECT(create_components(&c) == 0);
      EXPECT(start_pipeline(&c, DAMACY_F32, 2, 5, 2) == 0);
      struct damacy_sample samples[] = { sample(uri, 1, 2, 2, 5),
                                         sample(uri, 1, 2, 2, 5) };
      EXPECT(damacy_push(c.pipeline,
                         (struct damacy_sample_slice){ samples, samples + 2 })
               .status == DAMACY_OK);
      struct damacy_batch* batch = NULL;
      EXPECT(damacy_pop(c.pipeline, &batch) == DAMACY_OK);
      EXPECT(verify_crop(batch, -20, unsigned_bits[ti]) == 0);
      struct damacy_stats stats;
      damacy_stats_get(c.pipeline, &stats);
      EXPECT(stats.chunks_planned == 8);
      EXPECT(stats.chunks_dispatched == 4);
      EXPECT(stats.assemble.output_bytes == 2 * 2 * 5 * sizeof(float));
      EXPECT(stats.host_bytes_committed <= (32u << 20));
      EXPECT(stats.gpu_bytes_committed == 0);
      damacy_release(c.pipeline, batch);
      destroy_components(&c);
      fixture_rm_tree(root);
    }
  return 0;
}

static int
test_owned_plan(void)
{
  char root[] = "/tmp/damacy_owned_plan_XXXXXX";
  EXPECT(mkdtemp(root));
  char uri[256];
  snprintf(uri, sizeof(uri), "%s/array", root);
  int64_t shape[] = { 5, 11 }, chunks[] = { 2, 4 }, shards[] = { 4, 8 };
  EXPECT(fixture_write_zarr(uri, shape, chunks, shards, 2, "uint16", 10) == 0);
  struct components c = { 0 };
  EXPECT(create_components(&c) == 0);
  struct damacy_batch_spec output = { .dtype = DAMACY_F32,
                                      .sample_shape = { 2, 5 },
                                      .sample_rank = 2,
                                      .samples_per_batch = 2 };
  struct damacy_queue_limits queues = { .lookahead_samples = 4,
                                        .prepared_batches = 2 };
  EXPECT(c.planner->ops->start(c.planner, &output, &queues) == DAMACY_OK);
  struct prepared_plan* saved = NULL;
  for (unsigned i = 0; i < 12; ++i) {
    struct damacy_sample samples[] = { sample(uri, 1, 2, 2, 5),
                                       sample(uri, 1, 2, 2, 5) };
    EXPECT(
      c.planner->ops
        ->push(c.planner, (struct damacy_sample_slice){ samples, samples + 2 })
        .status == DAMACY_OK);
    struct prepared_plan* plan = NULL;
    enum damacy_status status = DAMACY_AGAIN;
    for (unsigned retry = 0; retry < 10000 && status == DAMACY_AGAIN; ++retry) {
      status = c.planner->ops->next(c.planner, &plan);
      if (status == DAMACY_AGAIN)
        platform_sleep_ns(1000000);
    }
    EXPECT(status == DAMACY_OK);
    if (!saved)
      saved = plan;
    else
      prepared_plan_destroy(plan);
    strcat(uri, "/.");
  }
  EXPECT(saved->n_arrays == 1 && saved->n_chunks == 4 && saved->n_uses == 8);
  damacy_planner_destroy(c.planner);
  c.planner = NULL;
  damacy_metadata_destroy(c.metadata);
  c.metadata = NULL;
  damacy_metadata_reader_destroy(c.metadata_reader);
  c.metadata_reader = NULL;
  EXPECT(saved->arrays[0].metadata.shape[1] == 11);
  struct damacy_stats stats = { 0 };
  EXPECT(c.executor->ops->start(c.executor, &output, &stats) == DAMACY_OK);
  EXPECT(c.executor->ops->submit(c.executor, saved, 42) == DAMACY_OK);
  struct damacy_batch* batch = NULL;
  for (unsigned retry = 0; retry < 10000 && !batch; ++retry) {
    int changed = 0;
    EXPECT(c.executor->ops->step(c.executor, &changed) == DAMACY_OK);
    enum damacy_status status = c.executor->ops->take(c.executor, &batch);
    EXPECT(status == DAMACY_OK || status == DAMACY_AGAIN);
    if (!batch)
      platform_sleep_ns(1000000);
  }
  EXPECT(batch);
  EXPECT(verify_crop(batch, 10, 16) == 0);
  damacy_batch_release(batch);
  destroy_components(&c);
  fixture_rm_tree(root);
  return 0;
}

static int
test_shared_metadata(void)
{
  char root[] = "/tmp/damacy_shared_metadata_XXXXXX";
  EXPECT(mkdtemp(root));
  char uri[256];
  snprintf(uri, sizeof(uri), "%s/array", root);
  int64_t shape[] = { 5, 11 }, chunks[] = { 2, 4 }, shards[] = { 4, 8 };
  EXPECT(fixture_write_zarr(uri, shape, chunks, shards, 2, "uint16", 10) == 0);
  struct components a = { 0 }, b = { 0 };
  EXPECT(create_components(&a) == 0);
  EXPECT(create_components(&b) == 0);
  damacy_planner_destroy(b.planner);
  b.planner = NULL;
  EXPECT(damacy_chunk_planner_create(
           a.metadata,
           &(struct damacy_plan_limits){ .max_chunks = 1024,
                                         .max_chunk_bytes = 1 << 20,
                                         .max_shards_per_sample = 4,
                                         .max_plan_bytes = 1 << 20 },
           &b.planner) == DAMACY_OK);
  EXPECT(start_pipeline(&a, DAMACY_F32, 2, 5, 2) == 0);
  EXPECT(start_pipeline(&b, DAMACY_F32, 2, 5, 2) == 0);
  damacy_metadata_reader_destroy(a.metadata_reader);
  a.metadata_reader = NULL;
  damacy_metadata_destroy(a.metadata);
  a.metadata = NULL;
  struct damacy* pipelines[] = { a.pipeline, b.pipeline };
  struct damacy_sample samples[] = { sample(uri, 1, 2, 2, 5),
                                     sample(uri, 1, 2, 2, 5) };
  for (unsigned i = 0; i < 2; ++i)
    EXPECT(damacy_push(pipelines[i],
                       (struct damacy_sample_slice){ samples, samples + 2 })
             .status == DAMACY_OK);
  for (unsigned i = 0; i < 2; ++i) {
    struct damacy_batch* batch = NULL;
    EXPECT(damacy_pop(pipelines[i], &batch) == DAMACY_OK);
    EXPECT(verify_crop(batch, 10, 16) == 0);
    damacy_release(pipelines[i], batch);
  }
  destroy_components(&a);
  destroy_components(&b);
  fixture_rm_tree(root);
  return 0;
}

struct pop_waiter
{
  struct damacy* pipeline;
  _Atomic int started;
  enum damacy_status status;
};

static void*
wait_pop(void* arg)
{
  struct pop_waiter* waiter = arg;
  atomic_store(&waiter->started, 1);
  struct damacy_batch* batch = NULL;
  waiter->status = damacy_pop(waiter->pipeline, &batch);
  damacy_batch_release(batch);
  return NULL;
}

static int
test_retained_outputs_and_shutdown(void)
{
  char root[] = "/tmp/damacy_lifetime_XXXXXX";
  EXPECT(mkdtemp(root));
  char uri[256];
  snprintf(uri, sizeof(uri), "%s/array", root);
  int64_t shape[] = { 5, 11 }, chunks[] = { 2, 4 }, shards[] = { 4, 8 };
  EXPECT(fixture_write_zarr(uri, shape, chunks, shards, 2, "uint16", 10) == 0);
  struct components c = { 0 };
  EXPECT(create_components(&c) == 0);
  EXPECT(start_pipeline(&c, DAMACY_F32, 2, 5, 2) == 0);
  struct damacy_sample samples[4];
  for (unsigned i = 0; i < 4; ++i)
    samples[i] = sample(uri, 1, 2, 2, 5);
  EXPECT(damacy_push(c.pipeline,
                     (struct damacy_sample_slice){ samples, samples + 4 })
           .status == DAMACY_OK);
  struct damacy_batch *first = NULL, *second = NULL;
  EXPECT(damacy_pop(c.pipeline, &first) == DAMACY_OK);
  EXPECT(damacy_pop(c.pipeline, &second) == DAMACY_OK);
  EXPECT(first != second);
  struct damacy_batch_info a, b;
  damacy_batch_info(first, &a);
  damacy_batch_info(second, &b);
  EXPECT(a.batch_id == 0 && b.batch_id == 1 && a.data != b.data);
  damacy_batch_retain(first);
  damacy_release(c.pipeline, first);
  EXPECT(damacy_push(c.pipeline,
                     (struct damacy_sample_slice){ samples, samples + 2 })
           .status == DAMACY_OK);
  struct pop_waiter waiter = { .pipeline = c.pipeline };
  pthread_t thread;
  EXPECT(pthread_create(&thread, NULL, wait_pop, &waiter) == 0);
  while (!atomic_load(&waiter.started))
    platform_sleep_ns(1000000);
  damacy_shutdown(c.pipeline);
  EXPECT(pthread_join(thread, NULL) == 0);
  EXPECT(waiter.status == DAMACY_SHUTDOWN);
  destroy_components(&c);
  EXPECT(verify_crop(first, 10, 16) == 0);
  EXPECT(verify_crop(second, 10, 16) == 0);
  damacy_batch_release(first);
  damacy_batch_release(second);
  fixture_rm_tree(root);
  return 0;
}

static int
test_bfloat_rounding_and_fill(void)
{
  char root[] = "/tmp/damacy_bfloat_XXXXXX";
  EXPECT(mkdtemp(root));
  char path[256];
  snprintf(path, sizeof(path), "%s/zarr.json", root);
  EXPECT(fixture_write_file(
           path,
           "{\"zarr_format\":3,\"node_type\":\"array\",\"shape\":[1,8],"
           "\"data_type\":\"float32\",\"fill_value\":-2,"
           "\"chunk_grid\":{\"name\":\"regular\",\"configuration\":{\"chunk_"
           "shape\":[1,8]}},"
           "\"chunk_key_encoding\":{\"name\":\"default\",\"configuration\":{"
           "\"separator\":\"/\"}},"
           "\"codecs\":[{\"name\":\"bytes\",\"configuration\":{\"endian\":"
           "\"little\"}}]}") == 0);
  snprintf(path, sizeof(path), "%s/c", root);
  EXPECT(mkdir(path, 0700) == 0);
  snprintf(path, sizeof(path), "%s/c/0", root);
  EXPECT(mkdir(path, 0700) == 0);
  snprintf(path, sizeof(path), "%s/c/0/0", root);
  uint32_t source[] = { 0x3f808000, 0x3f818000, 0xbf808000, 0x80000000,
                        0x7f800000, 0xff800000, 0x7fc00000, 0x00008000 };
  uint16_t expected[] = { 0x3f80, 0x3f82, 0xbf80, 0x8000,
                          0x7f80, 0xff80, 0x7fff, 0 };
  int fd = open(path, O_WRONLY | O_CREAT, 0600);
  EXPECT(fd >= 0 &&
         write(fd, source, sizeof(source)) == (ssize_t)sizeof(source));
  close(fd);
  for (unsigned missing = 0; missing < 2; ++missing) {
    struct components c = { 0 };
    EXPECT(create_components(&c) == 0);
    EXPECT(start_pipeline(&c, DAMACY_BF16, 1, 8, 1) == 0);
    struct damacy_sample request = sample(root, 0, 0, 1, 8);
    EXPECT(damacy_push(c.pipeline,
                       (struct damacy_sample_slice){ &request, &request + 1 })
             .status == DAMACY_OK);
    struct damacy_batch* batch = NULL;
    EXPECT(damacy_pop(c.pipeline, &batch) == DAMACY_OK);
    struct damacy_batch_info info;
    damacy_batch_info(batch, &info);
    for (unsigned i = 0; i < 8; ++i)
      EXPECT(((const uint16_t*)info.data)[i] ==
             (missing ? 0xc000 : expected[i]));
    damacy_batch_release(batch);
    destroy_components(&c);
    if (!missing)
      EXPECT(unlink(path) == 0);
  }
  fixture_rm_tree(root);
  return 0;
}

int
main(void)
{
  RUN(test_codecs_and_types);
  RUN(test_owned_plan);
  RUN(test_shared_metadata);
  RUN(test_retained_outputs_and_shutdown);
  RUN(test_bfloat_rounding_and_fill);
  return 0;
}
