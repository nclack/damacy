#include "damacy_pipeline.h"
#include "fixture.h"

#include <blosc.h>
#include <zstd.h>

#ifdef DAMACY_TEST_CUDA
#include <cuda_runtime.h>
#endif

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHUNK_ROWS 20
#define CHUNK_COLS 35
#define ARRAY_ROWS 40
#define ARRAY_COLS 70
#define CROP_ROWS 28
#define CROP_COLS 53
#define MAX_CHUNK_ROWS 512
#define ENCODED_CAP (MAX_CHUNK_ROWS * CHUNK_COLS * 8 + 256)

struct array_options
{
  const char* codec;
  unsigned typesize;
  int shuffle;
  int split;
  int level;
  int sharded;
  int raw_blocks;
  unsigned seed;
  unsigned chunk_rows;
};

#ifdef DAMACY_TEST_BLOSC_THREADS
static atomic_int wrong_blosc_threads;
int
__real_blosc_decompress_ctx(const void*, void*, size_t, int);

int
__wrap_blosc_decompress_ctx(const void* source,
                            void* dest,
                            size_t bytes,
                            int threads)
{
  if (threads != 1)
    atomic_store(&wrong_blosc_threads, 1);
  return __real_blosc_decompress_ctx(source, dest, bytes, threads);
}
#endif

static uint32_t
read_u32(const unsigned char* p)
{
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

static unsigned
chunk_rows(const struct array_options* options)
{
  return options->chunk_rows ? options->chunk_rows : CHUNK_ROWS;
}

static uint64_t
value_at(unsigned y, unsigned x, const struct array_options* options)
{
  uint64_t value = (y * ARRAY_COLS + x + options->seed) % 31;
  value *= 0x0102040810204081ull;
  unsigned chunk = (y / chunk_rows(options)) * 2 + x / CHUNK_COLS;
  if (options->raw_blocks == 1 ||
      (options->raw_blocks == 2 && chunk % 2 == 0)) {
    if (options->raw_blocks == 1 &&
        (y % chunk_rows(options)) * CHUNK_COLS + x % CHUNK_COLS >= 512) {
      value = 0;
    } else {
      value = y * ARRAY_COLS + x + options->seed + 1;
      value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
      value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
      value ^= value >> 31;
    }
  }
  if (options->typesize < 8)
    value &= (1ull << (8 * options->typesize)) - 1;
  return value;
}

static int
write_bytes(const char* path, const void* data, size_t size)
{
  FILE* file = fopen(path, "wb");
  EXPECT(file);
  EXPECT(fwrite(data, 1, size, file) == size);
  EXPECT(fclose(file) == 0);
  return 0;
}

static int
write_array(const char* uri, const struct array_options* options)
{
  const unsigned rows = chunk_rows(options);
  EXPECT(rows >= CHUNK_ROWS && rows <= MAX_CHUNK_ROWS);
  EXPECT(mkdir(uri, 0755) == 0);
  char path[512], codec[512], codecs[1024], json[2048];
  const char* shuffle = options->shuffle == BLOSC_SHUFFLE      ? "shuffle"
                        : options->shuffle == BLOSC_BITSHUFFLE ? "bitshuffle"
                                                               : "noshuffle";
  if (!strcmp(options->codec, "none"))
    codec[0] = 0;
  else if (!strcmp(options->codec, "zstd"))
    snprintf(codec,
             sizeof(codec),
             ",{\"name\":\"zstd\",\"configuration\":{\"level\":3}}");
  else
    snprintf(
      codec,
      sizeof(codec),
      ",{\"name\":\"blosc\",\"configuration\":{\"cname\":\"%s\","
      "\"clevel\":%d,\"shuffle\":\"%s\",\"typesize\":%u,\"blocksize\":%u}}",
      options->codec + 6,
      options->level,
      shuffle,
      options->typesize,
      256 * options->typesize);
  const char* bytes =
    "{\"name\":\"bytes\",\"configuration\":{\"endian\":\"little\"}}";
  if (options->sharded)
    snprintf(codecs,
             sizeof(codecs),
             "[{\"name\":\"sharding_indexed\",\"configuration\":{"
             "\"chunk_shape\":[%d,%d],\"codecs\":[%s%s],"
             "\"index_codecs\":[%s,{\"name\":\"crc32c\"}],\"index_location\":"
             "\"end\"}}]",
             rows,
             CHUNK_COLS,
             bytes,
             codec,
             bytes);
  else
    snprintf(codecs, sizeof(codecs), "[%s%s]", bytes, codec);
  snprintf(json,
           sizeof(json),
           "{\"zarr_format\":3,\"node_type\":\"array\",\"shape\":[%d,%d],"
           "\"data_type\":\"uint%u\",\"chunk_grid\":{\"name\":\"regular\","
           "\"configuration\":{\"chunk_shape\":[%d,%d]}},"
           "\"chunk_key_encoding\":{\"name\":\"default\",\"configuration\":{"
           "\"separator\":\"/\"}},"
           "\"fill_value\":0,\"codecs\":%s}",
           2 * rows,
           ARRAY_COLS,
           8 * options->typesize,
           options->sharded ? 2 * rows : rows,
           options->sharded ? ARRAY_COLS : CHUNK_COLS,
           codecs);
  snprintf(path, sizeof(path), "%s/zarr.json", uri);
  EXPECT(fixture_write_file(path, json) == 0);
  snprintf(path, sizeof(path), "%s/c", uri);
  EXPECT(mkdir(path, 0755) == 0);
  for (unsigned y = 0; y < (options->sharded ? 1u : 2u); ++y) {
    snprintf(path, sizeof(path), "%s/c/%u", uri, y);
    EXPECT(mkdir(path, 0755) == 0);
  }

  unsigned char encoded[4][ENCODED_CAP];
  uint64_t offsets[4], sizes[4];
  size_t total = 0;
  blosc_set_splitmode(options->split);
  for (unsigned chunk = 0; chunk < 4; ++chunk) {
    unsigned char source[MAX_CHUNK_ROWS * CHUNK_COLS * 8];
    size_t size = rows * CHUNK_COLS * options->typesize;
    for (unsigned y = 0; y < rows; ++y)
      for (unsigned x = 0; x < CHUNK_COLS; ++x) {
        uint64_t value = value_at(
          (chunk / 2) * rows + y, (chunk % 2) * CHUNK_COLS + x, options);
        for (unsigned byte = 0; byte < options->typesize; ++byte)
          source[(y * CHUNK_COLS + x) * options->typesize + byte] =
            (unsigned char)(value >> (8 * byte));
      }
    if (!strcmp(options->codec, "none")) {
      memcpy(encoded[chunk], source, size);
      sizes[chunk] = size;
    } else if (!strcmp(options->codec, "zstd")) {
      size_t n = ZSTD_compress(encoded[chunk], ENCODED_CAP, source, size, 3);
      EXPECT(!ZSTD_isError(n));
      sizes[chunk] = n;
    } else {
      int n = blosc_compress_ctx(options->level,
                                 options->shuffle,
                                 options->typesize,
                                 size,
                                 source,
                                 encoded[chunk],
                                 size + BLOSC_MAX_OVERHEAD,
                                 options->codec + 6,
                                 256 * options->typesize,
                                 1);
      EXPECT(n > 0);
      sizes[chunk] = (unsigned)n;
      uint32_t blocksize = read_u32(encoded[chunk] + 8);
      EXPECT(blocksize > 0 && blocksize <= size);
      if (options->chunk_rows)
        EXPECT(blocksize < size && size % blocksize != 0);
      EXPECT(
        !!(encoded[chunk][2] & BLOSC_MEMCPYED) ==
        (options->level == 0 || (options->raw_blocks == 2 && chunk % 2 == 0)));
      if (options->level && options->typesize > 1)
        EXPECT(!!(encoded[chunk][2] & 0x10) ==
               (options->split == BLOSC_NEVER_SPLIT));
    }
    offsets[chunk] = total;
    total += sizes[chunk];
    if (!options->sharded) {
      snprintf(path, sizeof(path), "%s/c/%u/%u", uri, chunk / 2, chunk % 2);
      EXPECT(write_bytes(path, encoded[chunk], sizes[chunk]) == 0);
    }
  }
  blosc_set_splitmode(BLOSC_FORWARD_COMPAT_SPLIT);
  if (options->sharded) {
    snprintf(path, sizeof(path), "%s/c/0/0", uri);
    EXPECT(fixture_write_synthetic_shard(path, total, offsets, sizes, 4) == 0);
    FILE* file = fopen(path, "r+b");
    EXPECT(file);
    for (unsigned chunk = 0; chunk < 4; ++chunk)
      EXPECT(fwrite(encoded[chunk], 1, sizes[chunk], file) == sizes[chunk]);
    EXPECT(fclose(file) == 0);
  }
  return 0;
}

static int
read_crops(const char* const* uris,
           const struct array_options* options,
           unsigned count,
           int gpu,
           unsigned workers,
           enum damacy_status expected)
{
  struct damacy_reader* reader = NULL;
  struct damacy_metadata_reader* metadata_reader = NULL;
  struct damacy_metadata* metadata = NULL;
  struct damacy_planner* planner = NULL;
  struct damacy_executor* executor = NULL;
  struct damacy* pipeline = NULL;
  EXPECT(count <= 32);
  EXPECT(damacy_file_reader_create(2, 8, &reader) == DAMACY_OK);
  EXPECT(damacy_file_metadata_reader_create(4, NULL, &metadata_reader) ==
         DAMACY_OK);
  EXPECT(
    damacy_zarr_metadata_create(metadata_reader,
                                &(struct damacy_metadata_cache_config){
                                  .array_entries = 64, .shard_entries = 256 },
                                &metadata) == DAMACY_OK);
  EXPECT(damacy_chunk_planner_create(
           metadata,
           &(struct damacy_plan_limits){ .max_chunks = 1024,
                                         .max_chunk_bytes = 1 << 18,
                                         .max_shards_per_sample = 4,
                                         .max_plan_bytes = 8 << 20 },
           &planner) == DAMACY_OK);
  if (gpu) {
    EXPECT(damacy_cuda_executor_create(
             reader,
             &(struct damacy_cuda_config){ .device = 0,
                                           .max_gpu_memory_bytes = 128 << 20,
                                           .max_chunk_bytes = 1 << 18,
                                           .max_read_bytes = 1 << 20,
                                           .max_chunks_per_wave = 64,
                                           .max_substreams_per_chunk = 32,
                                           .host_buffer_waves = 2,
                                           .chunk_layout_entries = 64 },
             &executor) == DAMACY_OK);
  } else {
    EXPECT(damacy_cpu_executor_create(
             reader,
             &(struct damacy_cpu_config){ .decode_workers = workers,
                                          .max_encoded_chunk_bytes = 1 << 18,
                                          .max_decoded_chunk_bytes = 1 << 18,
                                          .max_memory_bytes = 32 << 20,
                                          .chunks_per_input_buffer = 16 },
             &executor) == DAMACY_OK);
  }
  EXPECT(
    damacy_pipeline_create(
      planner,
      executor,
      &(struct damacy_batch_spec){ .dtype = DAMACY_U64,
                                   .sample_shape = { CROP_ROWS, CROP_COLS },
                                   .sample_rank = 2,
                                   .samples_per_batch = count },
      &(struct damacy_queue_limits){ .lookahead_samples = count,
                                     .prepared_batches = 2 },
      &pipeline) == DAMACY_OK);
  struct damacy_sample samples[32];
  for (unsigned i = 0; i < count; ++i)
    samples[i] = (struct damacy_sample){
      .uri = uris[i],
      .rank = 2,
      .axes = { { .kind = DAMACY_AXIS_INTERVAL,
                  .interval = { chunk_rows(&options[i]) - 14 + i % 2,
                                chunk_rows(&options[i]) - 14 + i % 2 +
                                  CROP_ROWS } },
                { .kind = DAMACY_AXIS_INTERVAL,
                  .interval = { 5 + i % 3, 5 + i % 3 + CROP_COLS } } }
    };
  EXPECT(damacy_push(pipeline,
                     (struct damacy_sample_slice){ samples, samples + count })
           .status == DAMACY_OK);
  struct damacy_batch* batch = NULL;
  enum damacy_status status = damacy_pop(pipeline, &batch);
  if (status != expected)
    log_error("codec=%s shuffle=%d split=%d bytes=%u sharded=%d gpu=%d: got %d "
              "expected %d",
              options[0].codec,
              options[0].shuffle,
              options[0].split,
              options[0].typesize,
              options[0].sharded,
              gpu,
              status,
              expected);
  EXPECT(status == expected);
  if (status == DAMACY_OK) {
    struct damacy_batch_info info;
    damacy_batch_info(batch, &info);
    size_t bytes = count * CROP_ROWS * CROP_COLS * sizeof(uint64_t);
    uint64_t* values = malloc(bytes);
    EXPECT(values);
    EXPECT(info.dtype == DAMACY_U64 && info.shape[0] == count);
#ifdef DAMACY_TEST_CUDA
    if (gpu)
      EXPECT(cudaMemcpy(values, info.data, bytes, cudaMemcpyDeviceToHost) ==
             cudaSuccess);
    else
#endif
      memcpy(values, info.data, bytes);
    for (unsigned i = 0; i < count; ++i)
      for (unsigned y = 0; y < CROP_ROWS; ++y)
        for (unsigned x = 0; x < CROP_COLS; ++x)
          EXPECT(values[(i * CROP_ROWS + y) * CROP_COLS + x] ==
                 value_at(chunk_rows(&options[i]) - 14 + i % 2 + y,
                          5 + i % 3 + x,
                          &options[i]));
    free(values);
    struct damacy_stats stats;
    damacy_stats_get(pipeline, &stats);
    if (gpu) {
      EXPECT(stats.gpu_bytes_committed <= (128ull << 20));
      if (count == 32)
        EXPECT(stats.waves_emitted >= 2);
    }
    damacy_release(pipeline, batch);
  } else {
    EXPECT(!batch);
  }
  damacy_destroy(pipeline);
  damacy_executor_destroy(executor);
  damacy_planner_destroy(planner);
  damacy_metadata_destroy(metadata);
  damacy_metadata_reader_destroy(metadata_reader);
  damacy_reader_destroy(reader);
  return 0;
}

static int
run_array(const struct array_options* options, int gpu)
{
  char root[] = "/tmp/damacy_lz4_XXXXXX", uri[256];
  EXPECT(mkdtemp(root));
  snprintf(uri, sizeof(uri), "%s/array", root);
  EXPECT(write_array(uri, options) == 0);
  const char* uris[] = { uri };
  EXPECT(read_crops(
           uris, options, 1, gpu, options->sharded ? 4 : 1, DAMACY_OK) == 0);
  fixture_rm_tree(root);
  return 0;
}

static int
test_layouts(int gpu)
{
  const char* codecs[] = { "blosc-lz4", "blosc-lz4hc" };
  for (unsigned codec = 0; codec < 2; ++codec)
    for (unsigned typesize = 1; typesize <= 8; typesize *= 2)
      for (int shuffle = 0; shuffle <= 2; ++shuffle)
        for (int split = BLOSC_ALWAYS_SPLIT; split <= BLOSC_NEVER_SPLIT;
             ++split)
          for (int sharded = 0; sharded <= 1; ++sharded) {
            struct array_options options = {
              codecs[codec], typesize, shuffle, split, 5, sharded, 0, 7, 0
            };
            EXPECT(run_array(&options, gpu) == 0);
          }
  return 0;
}

static int
test_raw_and_memcpyed(int gpu)
{
  for (int shuffle = 0; shuffle <= 2; ++shuffle)
    for (int sharded = 0; sharded <= 1; ++sharded) {
      struct array_options options = {
        "blosc-lz4", 8, shuffle, BLOSC_ALWAYS_SPLIT, 5, sharded, 1, 11, 0
      };
      EXPECT(run_array(&options, gpu) == 0);
      options.level = 0;
      EXPECT(run_array(&options, gpu) == 0);
      options.level = 5;
      options.raw_blocks = 2;
      EXPECT(run_array(&options, gpu) == 0);
    }
  return 0;
}

static int
test_split_tails(int gpu)
{
  const char* codecs[] = { "blosc-lz4", "blosc-lz4hc" };
  for (unsigned codec = 0; codec < 2; ++codec)
    for (unsigned rows = 480; rows <= 481; ++rows)
      for (int sharded = 0; sharded <= 1; ++sharded) {
        struct array_options options = { codecs[codec],
                                         8,
                                         BLOSC_BITSHUFFLE,
                                         BLOSC_ALWAYS_SPLIT,
                                         5,
                                         sharded,
                                         0,
                                         7,
                                         rows };
        EXPECT(run_array(&options, gpu) == 0);
      }
  return 0;
}

static int
test_mixed_and_growth(int gpu)
{
  char root[] = "/tmp/damacy_lz4_mixed_XXXXXX", paths[5][256];
  EXPECT(mkdtemp(root));
  const char* codecs[] = {
    "none", "zstd", "blosc-zstd", "blosc-lz4", "blosc-lz4hc"
  };
  const char* uris[32];
  struct array_options options[32];
  for (unsigned i = 0; i < 5; ++i) {
    options[i] = (struct array_options){
      codecs[i], 8, BLOSC_BITSHUFFLE, BLOSC_NEVER_SPLIT, 5, i % 2, 0, i, 0
    };
    if (i >= 3)
      options[i].split = BLOSC_ALWAYS_SPLIT;
    if (i == 4)
      options[i].chunk_rows = 480;
    snprintf(paths[i], sizeof(paths[i]), "%s/array%u", root, i);
    EXPECT(write_array(paths[i], &options[i]) == 0);
    uris[i] = paths[i];
  }
  EXPECT(read_crops(uris, options, 5, gpu, 2, DAMACY_OK) == 0);
  for (unsigned i = 0; i < 32; ++i) {
    uris[i] = paths[4];
    options[i] = (struct array_options){
      "blosc-lz4hc", 8, BLOSC_BITSHUFFLE, BLOSC_ALWAYS_SPLIT, 5, 0, 0, 4, 480
    };
  }
  EXPECT(read_crops(uris, options, 32, gpu, 4, DAMACY_OK) == 0);
  fixture_rm_tree(root);
  return 0;
}

static int
test_malformed(int gpu)
{
  for (unsigned error = 0; error < 13; ++error) {
    char root[] = "/tmp/damacy_lz4_bad_XXXXXX", uri[256], path[512];
    EXPECT(mkdtemp(root));
    snprintf(uri, sizeof(uri), "%s/array", root);
    struct array_options options = {
      "blosc-lz4", 2, BLOSC_BITSHUFFLE, BLOSC_NEVER_SPLIT, 5, 0, 0, 7, 0
    };
    EXPECT(write_array(uri, &options) == 0);
    snprintf(path, sizeof(path), "%s/c/1/1", uri);
    FILE* file = fopen(path, "rb");
    EXPECT(file);
    unsigned char data[ENCODED_CAP];
    size_t size = fread(data, 1, sizeof(data), file);
    EXPECT(size > 32 && !ferror(file));
    EXPECT(fclose(file) == 0);
    uint32_t block = read_u32(data + 16);
    EXPECT(block + 8 < size);
    switch (error) {
      case 0:
        size = 8;
        break;
      case 1:
        fixture_write_le32(data + 12, (uint32_t)size + 1);
        break;
      case 2:
        fixture_write_le32(data + 4, read_u32(data + 4) - 2);
        break;
      case 3:
        fixture_write_le32(data + 8, 0);
        break;
      case 4:
        data[3] = 1;
        break;
      case 5:
        data[2] = (data[2] & 0x1f) | 0x80;
        break;
      case 6:
        fixture_write_le32(data + 16, UINT32_MAX);
        break;
      case 7:
        fixture_write_le32(data + block, UINT32_MAX);
        break;
      case 8:
        fixture_write_le32(data + block, 0);
        break;
      case 9:
        fixture_write_le32(data + 12, (uint32_t)--size);
        break;
      case 10:
        fixture_write_le32(data + block, 2);
        data[block + 4] = 0x10;
        data[block + 5] = 0;
        break;
      case 11:
        fixture_write_le32(data + block, 3);
        memset(data + block + 4, 0, 3);
        break;
      case 12:
        data[0] = 3;
        break;
    }
    EXPECT(write_bytes(path, data, size) == 0);
    const char* uris[] = { uri };
    EXPECT(read_crops(uris, &options, 1, gpu, 2, DAMACY_DECODE) == 0);
    fixture_rm_tree(root);
  }
  return 0;
}

int
main(void)
{
  int backends = 1;
#ifdef DAMACY_TEST_CUDA
  backends = 2;
#endif
  for (int gpu = 0; gpu < backends; ++gpu) {
    EXPECT(test_layouts(gpu) == 0);
    EXPECT(test_raw_and_memcpyed(gpu) == 0);
    EXPECT(test_split_tails(gpu) == 0);
    EXPECT(test_mixed_and_growth(gpu) == 0);
    EXPECT(test_malformed(gpu) == 0);
  }
#ifdef DAMACY_TEST_BLOSC_THREADS
  EXPECT(!atomic_load(&wrong_blosc_threads));
#endif
  return 0;
}
