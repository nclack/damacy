// damacy_bench — exercises the public damacy API.
//
// Step 1 (current): stubs only. damacy_push always succeeds (no-op),
// damacy_pop returns DAMACY_AGAIN (no real batches yet). The bench
// measures call overhead and verifies the API shape compiles/links and
// behaves as documented. Real throughput numbers come back as the
// streaming pipeline lands (build-order steps 2–6).
//
//   ./damacy_bench [--batches N=200] [--batch-size N=8]
//                  [--lookahead N=4] [--io-threads N=8]
//                  [--host-buffer-mb N=256] [--device-buffer-mb N=512]
#include "damacy.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double
now_seconds(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

struct bench_args
{
  uint32_t n_batches;
  uint32_t batch_size;
  uint32_t lookahead_batches;
  uint32_t n_io_threads;
  uint64_t host_buffer_bytes;
  uint64_t device_buffer_bytes;
};

static void
usage(const char* argv0)
{
  fprintf(stderr,
          "usage: %s [opts]\n"
          "  --batches N           total batches to attempt (default 200)\n"
          "  --batch-size N        samples per batch (default 8)\n"
          "  --lookahead N         lookahead batches (default 4)\n"
          "  --io-threads N        io_queue worker threads (default 8)\n"
          "  --host-buffer-mb N    pinned staging total MB (default 256)\n"
          "  --device-buffer-mb N  device decompress total MB (default 512)\n",
          argv0);
}

static int
parse_args(int argc, char** argv, struct bench_args* out)
{
  *out = (struct bench_args){
    .n_batches = 200,
    .batch_size = 8,
    .lookahead_batches = 4,
    .n_io_threads = 8,
    .host_buffer_bytes = 256ull << 20,
    .device_buffer_bytes = 512ull << 20,
  };
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
#define NEXT()                                                                 \
  ((++i < argc) ? argv[i] : (usage(argv[0]), exit(1), (const char*)NULL))
    if (strcmp(a, "--batches") == 0)
      out->n_batches = (uint32_t)atoi(NEXT());
    else if (strcmp(a, "--batch-size") == 0)
      out->batch_size = (uint32_t)atoi(NEXT());
    else if (strcmp(a, "--lookahead") == 0)
      out->lookahead_batches = (uint32_t)atoi(NEXT());
    else if (strcmp(a, "--io-threads") == 0)
      out->n_io_threads = (uint32_t)atoi(NEXT());
    else if (strcmp(a, "--host-buffer-mb") == 0)
      out->host_buffer_bytes = (uint64_t)atoll(NEXT()) << 20;
    else if (strcmp(a, "--device-buffer-mb") == 0)
      out->device_buffer_bytes = (uint64_t)atoll(NEXT()) << 20;
    else {
      usage(argv[0]);
      return -1;
    }
#undef NEXT
  }
  return 0;
}

int
main(int argc, char** argv)
{
  struct bench_args args;
  if (parse_args(argc, argv, &args) != 0)
    return 1;

  struct damacy_config cfg = {
    .batch_size = args.batch_size,
    .lookahead_batches = args.lookahead_batches,
    .n_io_threads = args.n_io_threads,
    .host_buffer_bytes = args.host_buffer_bytes,
    .device_buffer_bytes = args.device_buffer_bytes,
    .n_zarrs_meta_cache = 4096,
    .n_shards_meta_cache = 16384,
    .dtype = DAMACY_U16,
  };

  struct damacy* d = NULL;
  enum damacy_status s = damacy_create(&cfg, &d);
  if (s != DAMACY_OK) {
    fprintf(stderr, "damacy_create: %s\n", damacy_status_str(s));
    return 1;
  }

  // Synthetic samples — the stub doesn't actually open them.
  const uint32_t n_samples = args.n_batches * args.batch_size;
  struct damacy_sample* samples =
    (struct damacy_sample*)calloc(n_samples, sizeof(*samples));
  if (!samples) {
    fprintf(stderr, "alloc samples failed\n");
    damacy_destroy(d);
    return 1;
  }
  for (uint32_t i = 0; i < n_samples; ++i) {
    samples[i].uri = "synthetic://stub";
    samples[i].aabb.rank = 5;
    for (uint8_t a = 0; a < 5; ++a)
      samples[i].aabb.dims[a] = (struct damacy_interval){ .beg = 0, .end = 1 };
  }

  double t0 = now_seconds();

  // Push everything in one slice; with stubs this consumes immediately.
  struct damacy_sample_slice slice = { .beg = samples,
                                       .end = samples + n_samples };
  struct damacy_push_result pr = damacy_push(d, slice);
  if (pr.status != DAMACY_OK) {
    fprintf(stderr,
            "damacy_push: %s (consumed %zu of %u)\n",
            damacy_status_str(pr.status),
            (size_t)(pr.unconsumed.beg - samples),
            n_samples);
    free(samples);
    damacy_destroy(d);
    return 1;
  }

  // Pop attempts. Stub returns AGAIN forever; we just count attempts.
  uint64_t pop_again = 0, pop_ok = 0;
  for (uint32_t i = 0; i < args.n_batches; ++i) {
    struct damacy_batch* b = NULL;
    enum damacy_status ps = damacy_pop(d, &b);
    if (ps == DAMACY_AGAIN) {
      ++pop_again;
    } else if (ps == DAMACY_OK) {
      ++pop_ok;
      damacy_release(d, b);
    } else {
      fprintf(stderr, "damacy_pop: %s\n", damacy_status_str(ps));
      break;
    }
  }

  enum damacy_status fs = damacy_flush(d);
  if (fs != DAMACY_OK)
    fprintf(stderr, "damacy_flush: %s\n", damacy_status_str(fs));

  double t1 = now_seconds();

  struct damacy_stats stats;
  damacy_stats_get(d, &stats);

  printf("samples pushed:         %u\n", n_samples);
  printf("pop attempts:           %u\n", args.n_batches);
  printf("pop OK:                 %llu\n", (unsigned long long)pop_ok);
  printf("pop AGAIN:              %llu\n", (unsigned long long)pop_again);
  printf("wall:                   %.3f s\n", t1 - t0);
  printf("batches_emitted:        %llu\n",
         (unsigned long long)stats.batches_emitted);
  printf("waves_emitted:          %llu\n",
         (unsigned long long)stats.waves_emitted);

  free(samples);
  damacy_destroy(d);
  return 0;
}
