#include "damacy_config.h"

#include "damacy_limits.h"
#include "log/log.h"
#include "platform/platform.h"
#include "util/prelude.h"

#include <math.h>
#include <string.h>

static const struct
{
  const char* name;
  const char* full_name;
  uint32_t bytes;
} output_types[] = {
  [DAMACY_F32] = { "f32", "float32", 4 },
  [DAMACY_BF16] = { "bf16", "bfloat16", 2 },
  [DAMACY_U8] = { "u8", "uint8", 1 },
  [DAMACY_U16] = { "u16", "uint16", 2 },
  [DAMACY_U32] = { "u32", "uint32", 4 },
  [DAMACY_U64] = { "u64", "uint64", 8 },
  [DAMACY_I8] = { "i8", "int8", 1 },
  [DAMACY_I16] = { "i16", "int16", 2 },
  [DAMACY_I32] = { "i32", "int32", 4 },
  [DAMACY_I64] = { "i64", "int64", 8 },
};

uint32_t
damacy_dtype_bpe(enum damacy_dtype dt)
{
  return (unsigned)dt < sizeof(output_types) / sizeof(output_types[0])
           ? output_types[dt].bytes
           : 0;
}

const char*
damacy_dtype_name(enum damacy_dtype dt)
{
  return damacy_dtype_bpe(dt) ? output_types[dt].name : "?";
}

int
damacy_dtype_from_string(const char* name,
                         size_t length,
                         enum damacy_dtype* out)
{
  for (unsigned i = 0; i < sizeof(output_types) / sizeof(output_types[0]);
       ++i) {
    if ((strlen(output_types[i].name) == length &&
         memcmp(name, output_types[i].name, length) == 0) ||
        (strlen(output_types[i].full_name) == length &&
         memcmp(name, output_types[i].full_name, length) == 0)) {
      *out = (enum damacy_dtype)i;
      return 0;
    }
  }
  return -1;
}

int
cast_path_supported(enum damacy_dtype dst, enum dtype src)
{
  if (!damacy_dtype_bpe(dst))
    return 0;
  switch (src) {
    case dtype_u8:
    case dtype_u16:
    case dtype_u32:
    case dtype_u64:
    case dtype_i8:
    case dtype_i16:
    case dtype_i32:
    case dtype_i64:
    case dtype_f16:
    case dtype_f32:
      return 1;
    default:
      return 0;
  }
}

enum damacy_status
validate_config(const struct damacy_config* cfg)
{
  CHECK_SILENT(Invalid, cfg);
  CHECK_SILENT(Invalid, cfg->samples_per_batch > 0);
  CHECK_SILENT(Invalid, cfg->tuning.max_gpu_memory_bytes > 0);
  CHECK_SILENT(Invalid,
               cfg->tuning.max_index_bytes <= UINT64_C(8) * UINT32_MAX);
  CHECK_SILENT(Invalid, cfg->lookahead_samples >= cfg->samples_per_batch);
  CHECK_SILENT(Invalid, cfg->tuning.n_io_threads > 0);
  CHECK_SILENT(Invalid, cfg->tuning.n_io_threads <= DAMACY_MAX_IO_THREADS);
  CHECK_SILENT(Invalid, cfg->tuning.metadata_io_concurrency > 0);
  CHECK_SILENT(Invalid,
               cfg->tuning.metadata_io_concurrency <=
                 DAMACY_MAX_METADATA_IO_CONCURRENCY);
  if (cfg->tuning.max_chunk_uncompressed_bytes == 0 ||
      cfg->tuning.max_chunk_uncompressed_bytes > DAMACY_MAX_CHUNK_BYTES) {
    log_error("max_chunk_uncompressed_bytes=%llu out of range (1..%llu)",
              (unsigned long long)cfg->tuning.max_chunk_uncompressed_bytes,
              (unsigned long long)DAMACY_MAX_CHUNK_BYTES);
    goto Invalid;
  }
  if (cfg->tuning.max_read_op_bytes == 0 ||
      cfg->tuning.max_read_op_bytes > UINT32_MAX) {
    log_error("max_read_op_bytes=%llu out of range (1..%llu)",
              (unsigned long long)cfg->tuning.max_read_op_bytes,
              (unsigned long long)UINT32_MAX);
    goto Invalid;
  }
  if (cfg->tuning.host_buffer_waves < DAMACY_N_WAVES ||
      cfg->tuning.host_buffer_waves > DAMACY_MAX_HOST_BUFFER_WAVES) {
    log_error("host_buffer_waves=%u out of range (%u..%u)",
              (unsigned)cfg->tuning.host_buffer_waves,
              (unsigned)DAMACY_N_WAVES,
              (unsigned)DAMACY_MAX_HOST_BUFFER_WAVES);
    goto Invalid;
  }
  if (cfg->tuning.max_chunks_per_wave == 0 ||
      cfg->tuning.max_chunks_per_wave > DAMACY_HARD_MAX_CHUNKS_PER_WAVE) {
    log_error("max_chunks_per_wave=%u out of range (1..%u)",
              (unsigned)cfg->tuning.max_chunks_per_wave,
              (unsigned)DAMACY_HARD_MAX_CHUNKS_PER_WAVE);
    goto Invalid;
  }
  if (cfg->tuning.max_substreams_per_chunk == 0 ||
      cfg->tuning.max_substreams_per_chunk >
        DAMACY_HARD_MAX_SUBSTREAMS_PER_CHUNK) {
    log_error("max_substreams_per_chunk=%u out of range (1..%u)",
              (unsigned)cfg->tuning.max_substreams_per_chunk,
              (unsigned)DAMACY_HARD_MAX_SUBSTREAMS_PER_CHUNK);
    goto Invalid;
  }
  CHECK_SILENT(Invalid, cfg->tuning.n_array_meta_cache > 0);
  CHECK_SILENT(Invalid, cfg->tuning.n_shard_index_cache > 0);
  CHECK_SILENT(Invalid, cfg->tuning.n_chunk_layout_cache > 0);
  CHECK_SILENT(Invalid, cfg->tuning.max_shards_per_sample > 0);
  // Metadata-cache floors: each cache pins every sample seq in
  // [watermark, pushed_samples). Push back-pressure bounds
  // pushed_samples - next_consume_seq <= lookahead_samples, and the watermark
  // lags next_consume_seq by the staging window (<= 2*samples_per_batch, the
  // planning_capacity cap), so the worst-case pinned set per cache is
  // lookahead_samples + 2*samples_per_batch. Sizing each cache to that floor
  // keeps pin-saturation out of normal operation; the cache also degrades to
  // DAMACY_AGAIN (retry) rather than crashing if it is ever hit. Messages
  // name the knob, observed vs required value, and the fix. See
  // dev/metadata_prefetch.md.
  {
    uint64_t meta_floor = (uint64_t)cfg->lookahead_samples +
                          2ull * (uint64_t)cfg->samples_per_batch;
    if ((uint64_t)cfg->tuning.n_array_meta_cache < meta_floor) {
      log_error("n_array_meta_cache=%u is too small: requires >= "
                "lookahead_samples(%u) + 2*samples_per_batch(%u) = %llu. Raise "
                "n_array_meta_cache to >= %llu.",
                (unsigned)cfg->tuning.n_array_meta_cache,
                (unsigned)cfg->lookahead_samples,
                (unsigned)cfg->samples_per_batch,
                (unsigned long long)meta_floor,
                (unsigned long long)meta_floor);
      goto Invalid;
    }
    if ((uint64_t)cfg->tuning.n_chunk_layout_cache < meta_floor) {
      log_error("n_chunk_layout_cache=%u is too small: requires >= "
                "lookahead_samples(%u) + 2*samples_per_batch(%u) = %llu. Raise "
                "n_chunk_layout_cache to >= %llu.",
                (unsigned)cfg->tuning.n_chunk_layout_cache,
                (unsigned)cfg->lookahead_samples,
                (unsigned)cfg->samples_per_batch,
                (unsigned long long)meta_floor,
                (unsigned long long)meta_floor);
      goto Invalid;
    }
    uint64_t shard_floor =
      meta_floor * (uint64_t)cfg->tuning.max_shards_per_sample;
    if ((uint64_t)cfg->tuning.n_shard_index_cache < shard_floor) {
      log_error(
        "n_shard_index_cache=%u is too small: requires >= "
        "(lookahead_samples(%u) + 2*samples_per_batch(%u)) * "
        "max_shards_per_sample(%u) = %llu. Raise n_shard_index_cache to >= "
        "%llu, or lower lookahead_samples / samples_per_batch / "
        "max_shards_per_sample.",
        (unsigned)cfg->tuning.n_shard_index_cache,
        (unsigned)cfg->lookahead_samples,
        (unsigned)cfg->samples_per_batch,
        (unsigned)cfg->tuning.max_shards_per_sample,
        (unsigned long long)shard_floor,
        (unsigned long long)shard_floor);
      goto Invalid;
    }
  }
  CHECK_SILENT(Invalid, damacy_dtype_bpe(cfg->dtype) > 0);
  CHECK_SILENT(Invalid, cfg->sample_rank > 0);
  CHECK_SILENT(Invalid, cfg->sample_rank <= DAMACY_MAX_RANK);
  for (uint8_t d = 0; d < cfg->sample_rank; ++d)
    CHECK_SILENT(Invalid, cfg->sample_shape[d] > 0);
  // AUTO=0 so designated-init callers get the default. PIN_TO requires
  // a non-negative node; AUTO and DISABLED ignore numa_node.
  CHECK_SILENT(Invalid,
               cfg->tuning.numa_strategy == DAMACY_NUMA_AUTO ||
                 cfg->tuning.numa_strategy == DAMACY_NUMA_DISABLED ||
                 cfg->tuning.numa_strategy == DAMACY_NUMA_PIN_TO);
  if (cfg->tuning.numa_strategy == DAMACY_NUMA_PIN_TO)
    CHECK_SILENT(Invalid, cfg->tuning.numa_node >= 0);
  CHECK_SILENT(Invalid,
               cfg->tuning.enable_gds == DAMACY_GDS_AUTO ||
                 cfg->tuning.enable_gds == DAMACY_GDS_ON ||
                 cfg->tuning.enable_gds == DAMACY_GDS_OFF);
  CHECK_SILENT(Invalid,
               isfinite(cfg->debug.metadata_latency.lognormal_mu_ln_ns));
  CHECK_SILENT(Invalid,
               isfinite(cfg->debug.metadata_latency.lognormal_sigma_ln_ns));
  CHECK_SILENT(Invalid,
               cfg->debug.metadata_latency.lognormal_sigma_ln_ns >= 0.0);
  return DAMACY_OK;
Invalid:
  return DAMACY_INVAL;
}

struct damacy_tuning
damacy_tuning_defaults(void)
{
  return (struct damacy_tuning){
    .max_chunk_uncompressed_bytes = 2 * 1024 * 1024,
    .max_read_op_bytes = 4 * 1024 * 1024,
    .max_index_bytes = 64ull << 20,
    .host_buffer_waves = DAMACY_DEFAULT_HOST_BUFFER_WAVES,
    .max_chunks_per_wave = DAMACY_DEFAULT_MAX_CHUNKS_PER_WAVE,
    .max_substreams_per_chunk = DAMACY_DEFAULT_MAX_SUBSTREAMS_PER_CHUNK,
    .n_io_threads = 64,
    .metadata_io_concurrency = 64,
    .n_array_meta_cache = DAMACY_DEFAULT_ARRAY_META_CACHE,
    .n_shard_index_cache = DAMACY_DEFAULT_SHARD_INDEX_CACHE,
    .n_chunk_layout_cache = DAMACY_DEFAULT_CHUNK_LAYOUT_CACHE,
    .max_shards_per_sample = DAMACY_DEFAULT_MAX_SHARDS_PER_SAMPLE,
    .numa_strategy = DAMACY_NUMA_AUTO,
    .enable_gds = DAMACY_GDS_AUTO,
  };
}

uint64_t
resolve_max_chunk_uncompressed(const struct damacy_config* cfg)
{
  return cfg->tuning.max_chunk_uncompressed_bytes;
}

uint64_t
resolve_max_read_op_bytes(const struct damacy_config* cfg)
{
  return cfg->tuning.max_read_op_bytes;
}

uint8_t
resolve_host_buffer_waves(const struct damacy_config* cfg)
{
  return cfg->tuning.host_buffer_waves;
}

uint32_t
resolve_max_chunks_per_wave(const struct damacy_config* cfg)
{
  return cfg->tuning.max_chunks_per_wave;
}

uint32_t
resolve_max_substreams_per_chunk(const struct damacy_config* cfg)
{
  return cfg->tuning.max_substreams_per_chunk;
}

uint32_t
resolve_metadata_io_concurrency(const struct damacy_config* cfg)
{
  return cfg->tuning.metadata_io_concurrency;
}

uint8_t
resolve_enable_gds(const struct damacy_config* cfg)
{
  if (cfg->tuning.enable_gds == DAMACY_GDS_ON)
    return 1;
  if (cfg->tuning.enable_gds == DAMACY_GDS_OFF)
    return 0;
  const char* e = platform_getenv("DAMACY_GDS_ENABLE");
  return (e && strcmp(e, "1") == 0) ? 1 : 0;
}

enum damacy_status
resolve_sample_shape(const struct damacy_config* cfg,
                     int64_t* out_shape,
                     uint8_t* out_rank)
{
  if (!cfg || !out_shape || !out_rank)
    return DAMACY_INVAL;
  if (cfg->sample_rank == 0 || cfg->sample_rank > DAMACY_MAX_RANK)
    return DAMACY_INVAL;
  for (uint8_t d = 0; d < cfg->sample_rank; ++d) {
    if (cfg->sample_shape[d] <= 0)
      return DAMACY_INVAL;
    out_shape[d] = cfg->sample_shape[d];
  }
  *out_rank = cfg->sample_rank;
  return DAMACY_OK;
}

enum damacy_status
resolve_sample_volume_bytes(const struct damacy_config* cfg,
                            uint64_t* out_bytes)
{
  if (!cfg || !out_bytes)
    return DAMACY_INVAL;
  int64_t shape[DAMACY_MAX_RANK];
  uint8_t rank = 0;
  enum damacy_status s = resolve_sample_shape(cfg, shape, &rank);
  if (s != DAMACY_OK)
    return s;
  uint32_t bpe = damacy_dtype_bpe(cfg->dtype);
  if (!bpe || !cfg->samples_per_batch)
    return DAMACY_INVAL;
  uint64_t volume = (uint64_t)cfg->samples_per_batch * bpe;
  for (uint8_t d = 0; d < rank; ++d) {
    if (volume > UINT64_MAX / (uint64_t)shape[d])
      return DAMACY_BUDGET;
    volume *= (uint64_t)shape[d];
  }
  *out_bytes = volume;
  return DAMACY_OK;
}
