// Regression for the gds_submit_dev partial-submit UAF: a bogus key
// mid-batch must drain the stream and release prior pins. Runs under
// cuFile compat mode when nvidia-fs is absent.

#include "cuda_init.h"
#include "damacy_stats.h"
#include "expect.h"
#include "pipeline/components.h"
#include "store/store.h"
#include "store/store_fs_gds.h"
#include "util/lru.h"

#include <cuda.h>
#include <errno.h>
#include <fcntl.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static unsigned advice_calls;
static int advice_error;
static int advised_fd;
static int invalid_advice;

int
__real_posix_fadvise(int fd, off_t offset, off_t len, int advice);

int
__wrap_posix_fadvise(int fd, off_t offset, off_t len, int advice)
{
  ++advice_calls;
  advised_fd = fd;
  invalid_advice |= offset != 0 || len != 0 || advice != POSIX_FADV_RANDOM;
  return advice_error ? advice_error
                      : __real_posix_fadvise(fd, offset, len, advice);
}

static int
write_byte(const char* path, char c)
{
  FILE* f = fopen(path, "wb");
  if (!f)
    return 1;
  size_t n = fwrite(&c, 1, 1, f);
  fclose(f);
  return n == 1 ? 0 : 1;
}

static int
test_submit_fail_releases_pins(void)
{
  if (cuda_init_primary()) {
    log_info("test_store_fs_gds: no CUDA device; skipping");
    return 0;
  }

  char root[] = "/tmp/damacy_store_fs_gds_XXXXXX";
  EXPECT(mkdtemp(root));
  for (int i = 0; i < 3; ++i) {
    char p[256];
    snprintf(p, sizeof p, "%s/k%d", root, i);
    EXPECT(write_byte(p, (char)('a' + i)) == 0);
  }

  struct store_fs_gds_config cfg = {
    .root = root,
    .fd_cache_capacity = 8,
  };
  struct store* s = store_fs_gds_create(&cfg);
  if (!s) {
    log_info(
      "test_store_fs_gds: store_fs_gds_create returned NULL (no cuFile); "
      "skipping");
    return 0;
  }

  CUstream stream = NULL;
  EXPECT(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS);
  store_fs_gds_set_stream(s, stream);

  CUdeviceptr dbuf = 0;
  EXPECT(cuMemAlloc(&dbuf, 4096) == CUDA_SUCCESS);

  struct store_read reads[] = {
    { .key = "k0", .dst = (void*)dbuf, .offset = 0, .len = 1 },
    { .key = "k1", .dst = (void*)(dbuf + 16), .offset = 0, .len = 1 },
    { .key = "missing_no_such_file",
      .dst = (void*)(dbuf + 32),
      .offset = 0,
      .len = 1 },
    { .key = "k2", .dst = (void*)(dbuf + 48), .offset = 0, .len = 1 },
  };
  struct store_submit_result submit = store_read_submit_dev(s, reads, 4);
  EXPECT(submit.status == DAMACY_IO);

  struct lru_stats stats;
  store_fs_gds_stats_get(s, &stats);
  EXPECT(stats.pinned == 0);

  struct store_read good[] = {
    { .key = "k0", .dst = (void*)dbuf, .offset = 0, .len = 1 },
  };
  struct store_submit_result submit2 = store_read_submit_dev(s, good, 1);
  EXPECT(submit2.status == DAMACY_OK);
  EXPECT(cuStreamSynchronize(stream) == CUDA_SUCCESS);
  store_event_discard(s, submit2.event);

  cuMemFree(dbuf);
  store_destroy(s);
  cuStreamDestroy(stream);
  for (int i = 0; i < 3; ++i) {
    char p[256];
    snprintf(p, sizeof p, "%s/k%d", root, i);
    unlink(p);
  }
  rmdir(root);
  return 0;
}

static int
write_n(const char* path, size_t n)
{
  FILE* f = fopen(path, "wb");
  if (!f)
    return 1;
  char* buf = (char*)calloc(1, n);
  if (!buf) {
    fclose(f);
    return 1;
  }
  for (size_t i = 0; i < n; ++i)
    buf[i] = (char)(i & 0xff);
  size_t w = fwrite(buf, 1, n, f);
  free(buf);
  fclose(f);
  return w == n ? 0 : 1;
}

static int
read_byte(struct store* store,
          CUdeviceptr buffer,
          const char* key,
          char expected)
{
  struct store_read read = { .key = key, .dst = (void*)buffer, .len = 1 };
  struct store_submit_result submit = store_read_submit_dev(store, &read, 1);
  EXPECT(submit.status == DAMACY_OK);
  EXPECT(store_event_wait(store, submit.event) == DAMACY_OK);
  char value = 0;
  EXPECT(cuMemcpyDtoH(&value, buffer, 1) == CUDA_SUCCESS);
  EXPECT(value == expected);
  return 0;
}

static int
test_readahead_cache_and_retry(void)
{
  if (cuda_init_primary()) {
    log_info("test_store_fs_gds: no CUDA device; skipping");
    return 0;
  }
  char root[] = "/tmp/damacy_gds_readahead_XXXXXX";
  EXPECT(mkdtemp(root));
  char paths[2][256];
  for (int i = 0; i < 2; ++i) {
    snprintf(paths[i], sizeof(paths[i]), "%s/k%d", root, i);
    EXPECT(write_byte(paths[i], (char)('a' + i)) == 0);
  }
  CUstream stream = NULL;
  CUdeviceptr buffer = 0;
  EXPECT(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&buffer, 4096) == CUDA_SUCCESS);
  for (int disabled = 0; disabled <= 1; ++disabled) {
    struct store* store = store_fs_gds_create(&(struct store_fs_gds_config){
      .root = root, .fd_cache_capacity = 1, .disable_readahead = disabled });
    EXPECT(store);
    store_fs_gds_set_stream(store, stream);
    for (int pass = 0; pass < 3; ++pass) {
      const char* key = pass == 1 ? "k1" : "k0";
      unsigned before = advice_calls;
      EXPECT(read_byte(store, buffer, key, pass == 1 ? 'b' : 'a') == 0);
      EXPECT(advice_calls == before + disabled);
      before = advice_calls;
      EXPECT(read_byte(store, buffer, key, pass == 1 ? 'b' : 'a') == 0);
      EXPECT(advice_calls == before);
    }
    if (disabled) {
      advice_error = EIO;
      struct store_read read = { .key = "k1", .dst = (void*)buffer, .len = 1 };
      unsigned before = advice_calls;
      EXPECT(store_read_submit_dev(store, &read, 1).status == DAMACY_IO);
      EXPECT(advice_calls == before + 1);
      EXPECT(fcntl(advised_fd, F_GETFD) == -1 && errno == EBADF);
      struct lru_stats stats;
      store_fs_gds_stats_get(store, &stats);
      EXPECT(stats.pinned == 0);
      advice_error = 0;
      EXPECT(read_byte(store, buffer, "k1", 'b') == 0);
      EXPECT(advice_calls == before + 2);
    }
    store_destroy(store);
  }
  EXPECT(cuMemFree(buffer) == CUDA_SUCCESS);
  EXPECT(cuStreamDestroy(stream) == CUDA_SUCCESS);
  for (int i = 0; i < 2; ++i)
    EXPECT(unlink(paths[i]) == 0);
  EXPECT(rmdir(root) == 0);
  return 0;
}

// A raw payload needs no header reads through the ordinary reader store, so
// these advice calls must come from the GDS store created by the executor.
static int
test_cuda_reader_readahead(void)
{
  if (cuda_init_primary()) {
    log_info("test_store_fs_gds: no CUDA device; skipping");
    return 0;
  }
  char path[] = "/tmp/damacy_gds_reader_XXXXXX";
  int fd = mkstemp(path);
  EXPECT(fd >= 0);
  EXPECT(close(fd) == 0);
  EXPECT(write_n(path, 4096) == 0);
  const struct damacy_batch_spec output = { .dtype = DAMACY_U8,
                                            .sample_shape = { 1 },
                                            .sample_rank = 1,
                                            .samples_per_batch = 1 };
  struct plan_array array = { .uri = path,
                              .metadata = {
                                .rank = 1,
                                .dtype = dtype_u8,
                                .shape = { 4096 },
                                .inner_chunk_shape = { 4096 },
                                .inner_codec = { .id = CODEC_NONE } } };
  struct plan_chunk chunk = { .path = path,
                              .encoded_bytes = 4096,
                              .decoded_bytes = 4096 };
  struct plan_region region = { .source = { .rank = 1, .dims = { { 1, 2 } } } };
  struct plan_use use = { .next = UINT32_MAX };
  for (uint8_t enabled = 0; enabled <= 1; ++enabled) {
    struct damacy_reader* reader = NULL;
    EXPECT(damacy_file_reader_create_with_config(
             &(struct damacy_file_reader_config){ .workers = 1,
                                                  .max_inflight_reads = 4,
                                                  .enable_readahead = enabled },
             &reader) == DAMACY_OK);
    struct damacy_executor* executor = NULL;
    EXPECT(damacy_cuda_executor_create(reader,
                                       &(struct damacy_cuda_config){
                                         .device = 0,
                                         .max_gpu_memory_bytes = 64 << 20,
                                         .max_chunk_bytes = 4096,
                                         .max_read_bytes = 4096,
                                         .max_chunks_per_wave = 1,
                                         .max_substreams_per_chunk = 1,
                                         .host_buffer_waves = 2,
                                         .chunk_layout_entries = 1,
                                         .numa_strategy = DAMACY_NUMA_DISABLED,
                                         .enable_gds = DAMACY_GDS_ON },
                                       &executor) == DAMACY_OK);
    struct damacy_stats stats;
    stats_init(&stats);
    EXPECT(executor->ops->start(executor, &output, &stats) == DAMACY_OK);
    struct prepared_plan* plan = calloc(1, sizeof(*plan));
    EXPECT(plan);
    *plan = (struct prepared_plan){ .output = output,
                                    .arrays = &array,
                                    .chunks = &chunk,
                                    .regions = &region,
                                    .uses = &use,
                                    .n_arrays = 1,
                                    .n_chunks = 1,
                                    .n_regions = 1,
                                    .n_uses = 1 };
    unsigned before = advice_calls;
    EXPECT(executor->ops->submit(executor, plan, 0) == DAMACY_OK);
    EXPECT(executor->ops->enter_thread(executor) == DAMACY_OK);
    struct damacy_batch* batch = NULL;
    for (unsigned i = 0; i < 10000 && !batch; ++i) {
      int changed = 0;
      EXPECT(executor->ops->step(executor, &changed) == DAMACY_OK);
      enum damacy_status status = executor->ops->take(executor, &batch);
      EXPECT(status == DAMACY_OK || status == DAMACY_AGAIN);
      if (!batch)
        platform_sleep_ns(1000000);
    }
    EXPECT(batch);
    EXPECT(advice_calls == before + !enabled);
    struct damacy_batch_info info;
    damacy_batch_info(batch, &info);
    EXPECT(cuStreamSynchronize((CUstream)info.ready_stream) == CUDA_SUCCESS);
    uint8_t value = 0;
    EXPECT(cuMemcpyDtoH(&value, (CUdeviceptr)info.data, 1) == CUDA_SUCCESS);
    EXPECT(value == 1);
    damacy_batch_release(batch);
    executor->ops->leave_thread(executor);
    damacy_executor_destroy(executor);
    damacy_reader_destroy(reader);
  }
  EXPECT(unlink(path) == 0);
  return 0;
}

// cuStreamWaitValue32 gates the stream from CPU memory; a spinning
// host callback would risk deadlock against fs_gds_free_params_cb on
// driver pools sized to one.
static int
test_event_query_reflects_completion(void)
{
  if (cuda_init_primary()) {
    log_info("test_store_fs_gds: no CUDA device; skipping");
    return 0;
  }

  enum
  {
    PAYLOAD = 16u * 1024u * 1024u,
  };

  char root[] = "/tmp/damacy_store_fs_gds_eq_XXXXXX";
  EXPECT(mkdtemp(root));

  char path[256];
  snprintf(path, sizeof path, "%s/blob", root);
  EXPECT(write_n(path, PAYLOAD) == 0);

  struct store_fs_gds_config cfg = {
    .root = root,
    .fd_cache_capacity = 8,
  };
  struct store* s = store_fs_gds_create(&cfg);
  if (!s) {
    log_info(
      "test_store_fs_gds: store_fs_gds_create returned NULL (no cuFile); "
      "skipping");
    return 0;
  }

  CUstream stream = NULL;
  EXPECT(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS);
  store_fs_gds_set_stream(s, stream);

  CUdeviceptr dbuf = 0;
  EXPECT(cuMemAlloc(&dbuf, PAYLOAD) == CUDA_SUCCESS);

  uint32_t* gate_host = NULL;
  EXPECT(cuMemHostAlloc((void**)&gate_host,
                        sizeof(*gate_host),
                        CU_MEMHOSTALLOC_DEVICEMAP) == CUDA_SUCCESS);
  *gate_host = 0;
  CUdeviceptr gate_dev = 0;
  EXPECT(cuMemHostGetDevicePointer(&gate_dev, gate_host, 0) == CUDA_SUCCESS);
  EXPECT(cuStreamWaitValue32(stream, gate_dev, 1u, CU_STREAM_WAIT_VALUE_EQ) ==
         CUDA_SUCCESS);

  struct store_read read = {
    .key = "blob",
    .dst = (void*)dbuf,
    .offset = 0,
    .len = PAYLOAD,
  };
  struct store_submit_result submit = store_read_submit_dev(s, &read, 1);
  EXPECT(submit.status == DAMACY_OK);

  struct store_event_poll poll = store_event_query(s, submit.event);
  EXPECT(!poll.ready);

  atomic_store_explicit((_Atomic uint32_t*)gate_host, 1u, memory_order_release);
  EXPECT(cuStreamSynchronize(stream) == CUDA_SUCCESS);

  poll = store_event_query(s, submit.event);
  EXPECT(poll.ready);
  EXPECT(poll.status == DAMACY_OK);

  cuMemFreeHost(gate_host);
  cuMemFree(dbuf);
  store_destroy(s);
  cuStreamDestroy(stream);
  unlink(path);
  rmdir(root);
  return 0;
}

// cuFile docs require cuFileStreamDeregister before cuStreamDestroy; the
// post-destroy cuStreamDestroy below relies on libcufile returning an
// error otherwise.
static int
test_set_stream_replace_and_destroy(void)
{
  if (cuda_init_primary()) {
    log_info("test_store_fs_gds: no CUDA device; skipping");
    return 0;
  }

  char root[] = "/tmp/damacy_store_fs_gds_XXXXXX";
  EXPECT(mkdtemp(root));

  struct store_fs_gds_config cfg = {
    .root = root,
    .fd_cache_capacity = 8,
  };
  struct store* s = store_fs_gds_create(&cfg);
  if (!s) {
    log_info(
      "test_store_fs_gds: store_fs_gds_create returned NULL (no cuFile); "
      "skipping");
    return 0;
  }

  CUstream a = NULL;
  CUstream b = NULL;
  EXPECT(cuStreamCreate(&a, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS);
  EXPECT(cuStreamCreate(&b, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS);

  store_fs_gds_set_stream(s, a);
  store_fs_gds_set_stream(s, a);
  store_fs_gds_set_stream(s, b);
  store_fs_gds_set_stream(s, NULL);

  store_destroy(s);

  EXPECT(cuStreamDestroy(a) == CUDA_SUCCESS);
  EXPECT(cuStreamDestroy(b) == CUDA_SUCCESS);
  return 0;
}

static int
test_destroy_deregisters_active_stream(void)
{
  if (cuda_init_primary()) {
    log_info("test_store_fs_gds: no CUDA device; skipping");
    return 0;
  }

  char root[] = "/tmp/damacy_store_fs_gds_XXXXXX";
  EXPECT(mkdtemp(root));

  struct store_fs_gds_config cfg = {
    .root = root,
    .fd_cache_capacity = 8,
  };
  struct store* s = store_fs_gds_create(&cfg);
  if (!s) {
    log_info(
      "test_store_fs_gds: store_fs_gds_create returned NULL (no cuFile); "
      "skipping");
    return 0;
  }

  CUstream a = NULL;
  EXPECT(cuStreamCreate(&a, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS);
  store_fs_gds_set_stream(s, a);
  store_destroy(s);
  EXPECT(cuStreamDestroy(a) == CUDA_SUCCESS);
  return 0;
}

int
main(void)
{
  // Must be set before cuFile init; lets the test run without nvidia-fs.
  setenv("CUFILE_FORCE_COMPAT_MODE", "true", 1);
  if (cuda_init_primary()) {
    log_info("test_store_fs_gds: no CUDA device; skipping");
    return 77;
  }
  struct store* probe =
    store_fs_gds_create(&(struct store_fs_gds_config){ .root = "" });
  if (!probe) {
    log_info("test_store_fs_gds: no cuFile; skipping");
    return 77;
  }
  store_destroy(probe);
  RUN(test_submit_fail_releases_pins);
  RUN(test_readahead_cache_and_retry);
  RUN(test_cuda_reader_readahead);
  RUN(test_event_query_reflects_completion);
  RUN(test_set_stream_replace_and_destroy);
  RUN(test_destroy_deregisters_active_stream);
  EXPECT(!invalid_advice);
  return 0;
}
