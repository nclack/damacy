// Throughput bench driver. Walks the pipeline for `--batches` batches and
// prints read GB/s, decode GB/s, and end-to-end GB/s.
//
//   ./damacy_bench --store <path>
//                  [--batch-size N=8]
//                  [--prefetch D=2]
//                  [--chunks-per-sample C=8]
//                  [--batches N=200]
//                  [--pattern sequential|random]
//                  [--io-threads T=8]
//                  [--device 0]
#include "decoder.h"
#include "pipeline.h"
#include "store.h"
#include "zarr.h"

#include <cuda_runtime.h>

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

static void
usage(const char* argv0)
{
  fprintf(
    stderr,
    "usage: %s --store <path> [opts]\n"
    "  --store <path>             root of the zarr store (required)\n"
    "  --array <prefix>           array prefix within the store (default "
    "\"\")\n"
    "  --batch-size N             samples per batch (default 8)\n"
    "  --prefetch D               batches in flight (default 2)\n"
    "  --chunks-per-sample C      contiguous chunks per sample (default 8)\n"
    "  --batches N                number of batches (default 200)\n"
    "  --pattern P                sequential | random (default sequential)\n"
    "  --io-threads T             store io_queue threads (default 8)\n"
    "  --device D                 CUDA device id (default 0)\n",
    argv0);
}

struct bench_args
{
  const char* store_root;
  const char* array_prefix;
  int batch_size;
  int prefetch;
  int chunks_per_sample;
  int n_batches;
  int io_threads;
  int device_id;
  enum pipeline_pattern pattern;
};

static int
parse_args(int argc, char** argv, struct bench_args* out)
{
  *out = (struct bench_args){
    .store_root = NULL,
    .array_prefix = "",
    .batch_size = 8,
    .prefetch = 2,
    .chunks_per_sample = 8,
    .n_batches = 200,
    .io_threads = 8,
    .device_id = 0,
    .pattern = PIPELINE_PATTERN_SEQUENTIAL,
  };

  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
#define NEXT()                                                                 \
  ((++i < argc) ? argv[i] : (usage(argv[0]), exit(1), (const char*)NULL))
    if (strcmp(a, "--store") == 0)
      out->store_root = NEXT();
    else if (strcmp(a, "--array") == 0)
      out->array_prefix = NEXT();
    else if (strcmp(a, "--batch-size") == 0)
      out->batch_size = atoi(NEXT());
    else if (strcmp(a, "--prefetch") == 0)
      out->prefetch = atoi(NEXT());
    else if (strcmp(a, "--chunks-per-sample") == 0)
      out->chunks_per_sample = atoi(NEXT());
    else if (strcmp(a, "--batches") == 0)
      out->n_batches = atoi(NEXT());
    else if (strcmp(a, "--io-threads") == 0)
      out->io_threads = atoi(NEXT());
    else if (strcmp(a, "--device") == 0)
      out->device_id = atoi(NEXT());
    else if (strcmp(a, "--pattern") == 0) {
      const char* v = NEXT();
      if (strcmp(v, "random") == 0)
        out->pattern = PIPELINE_PATTERN_RANDOM;
      else if (strcmp(v, "sequential") == 0)
        out->pattern = PIPELINE_PATTERN_SEQUENTIAL;
      else {
        usage(argv[0]);
        return -1;
      }
    } else {
      usage(argv[0]);
      return -1;
    }
#undef NEXT
  }
  if (!out->store_root) {
    usage(argv[0]);
    return -1;
  }
  return 0;
}

int
main(int argc, char** argv)
{
  struct bench_args args;
  if (parse_args(argc, argv, &args) != 0)
    return 1;

  struct store_fs_config sc = {
    .root = args.store_root,
    .nthreads = args.io_threads,
  };
  struct store* store = store_fs_create(&sc);
  if (!store) {
    fprintf(stderr, "store_fs_create failed\n");
    return 1;
  }

  struct zarr_reader_config rc = { .store = store,
                                   .prefix = args.array_prefix };
  struct zarr_reader* reader = zarr_reader_open(&rc);
  if (!reader) {
    fprintf(stderr,
            "zarr_reader_open failed (check that %s/%szarr.json exists "
            "and the array uses sharded zstd)\n",
            args.store_root,
            args.array_prefix[0] ? args.array_prefix : "");
    store_destroy(store);
    return 1;
  }

  const struct zarr_array_info* info = zarr_reader_info(reader);
  size_t inner_uncompressed = zarr_reader_chunk_uncompressed_bytes(reader);
  fprintf(stderr,
          "array: rank=%u dtype=%d inner_uncompressed=%zu bytes\n",
          info->rank,
          (int)info->dtype,
          inner_uncompressed);
  for (uint8_t d = 0; d < info->rank; ++d) {
    fprintf(stderr,
            "  dim[%u]: size=%llu chunk_size=%llu chunks_per_shard=%llu\n",
            d,
            (unsigned long long)info->dims[d].size,
            (unsigned long long)info->dims[d].chunk_size,
            (unsigned long long)info->dims[d].chunks_per_shard);
  }

  int chunks_per_batch = args.batch_size * args.chunks_per_sample;
  size_t max_compressed = inner_uncompressed * 2;

  struct decoder* dec = decoder_create(
    args.device_id, (size_t)chunks_per_batch, inner_uncompressed);
  if (!dec) {
    fprintf(stderr, "decoder_create failed\n");
    zarr_reader_close(reader);
    store_destroy(store);
    return 1;
  }

  struct pipeline_config pc = {
    .store = store,
    .reader = reader,
    .decoder = dec,
    .batch_size = args.batch_size,
    .chunks_per_sample = args.chunks_per_sample,
    .prefetch_depth = args.prefetch,
    .device_id = args.device_id,
    .pattern = args.pattern,
    .seed = 0xC0FFEEull,
    .max_compressed_chunk_bytes = max_compressed,
  };
  struct pipeline* p = pipeline_create(&pc);
  if (!p) {
    fprintf(stderr, "pipeline_create failed\n");
    decoder_destroy(dec);
    zarr_reader_close(reader);
    store_destroy(store);
    return 1;
  }

  // Warm-up batch (not counted) to load shard indices and prime caches.
  {
    struct pipeline_batch wb;
    if (pipeline_next(p, &wb) == 0)
      pipeline_release(p, wb.batch_id);
  }

  double t0 = now_seconds();
  for (int i = 0; i < args.n_batches; ++i) {
    struct pipeline_batch b;
    if (pipeline_next(p, &b)) {
      fprintf(stderr, "pipeline_next failed at batch %d\n", i);
      break;
    }
    pipeline_release(p, b.batch_id);
  }
  double t1 = now_seconds();
  double wall = t1 - t0;

  struct pipeline_stats s;
  pipeline_stats_get(p, &s);
  // Subtract the warm-up batch from cumulative counters.
  // (Stats include the warm-up; correct for it.)
  uint64_t batches_for_stats = s.n_batches > 0 ? s.n_batches - 1 : 0;
  uint64_t comp_bytes = s.bytes_read_compressed;
  uint64_t dec_bytes = s.bytes_decompressed;
  // Estimate the warm-up's share by averaging.
  if (batches_for_stats > 0 && s.n_batches > 0) {
    comp_bytes -= comp_bytes / s.n_batches;
    dec_bytes -= dec_bytes / s.n_batches;
  }

  double comp_gbps = (comp_bytes / 1e9) / wall;
  double dec_gbps = (dec_bytes / 1e9) / wall;

  printf("batches:                %llu\n",
         (unsigned long long)batches_for_stats);
  printf("wall:                   %.3f s\n", wall);
  printf("compressed bytes read:  %.3f GB\n", comp_bytes / 1e9);
  printf("decompressed bytes:     %.3f GB\n", dec_bytes / 1e9);
  printf("compressed read:        %.3f GB/s\n", comp_gbps);
  printf("decompressed throughput:%.3f GB/s\n", dec_gbps);

  pipeline_destroy(p);
  decoder_destroy(dec);
  zarr_reader_close(reader);
  store_destroy(store);
  return 0;
}
