#include "damacy_config.h"

#include "damacy_limits.h"
#include "platform/platform.h"
#include "util/prelude.h"

#include <string.h>

uint32_t
damacy_dtype_bpe(enum damacy_dtype dt)
{
  switch (dt) {
    case DAMACY_BF16:
      return 2;
    case DAMACY_F32:
      return 4;
  }
  return 0;
}

int
cast_path_supported(enum damacy_dtype dst, enum dtype src)
{
  switch (dst) {
    case DAMACY_F32:
    case DAMACY_BF16:
      switch (src) {
        case dtype_u8:
        case dtype_u16:
        case dtype_i16:
        case dtype_u32:
        case dtype_i32:
        case dtype_f16:
        case dtype_f32:
          return 1;
        default:
          return 0;
      }
  }
  return 0;
}

enum damacy_status
validate_config(const struct damacy_config* cfg)
{
  CHECK_SILENT(Invalid, cfg);
  CHECK_SILENT(Invalid, cfg->batch_size > 0);
  CHECK_SILENT(Invalid, cfg->tuning.max_gpu_memory_bytes > 0);
  CHECK_SILENT(Invalid, cfg->lookahead_batches >= 2);
  CHECK_SILENT(Invalid, cfg->tuning.n_io_threads > 0);
  CHECK_SILENT(Invalid, cfg->tuning.n_io_threads <= DAMACY_MAX_IO_THREADS);
  CHECK_SILENT(
    Invalid,
    cfg->tuning.host_buffer_waves == 0 ||
      (cfg->tuning.host_buffer_waves >= DAMACY_N_WAVES &&
       cfg->tuning.host_buffer_waves <= DAMACY_MAX_HOST_BUFFER_WAVES));
  CHECK_SILENT(Invalid, cfg->tuning.n_zarrs_meta_cache > 0);
  CHECK_SILENT(Invalid, cfg->tuning.n_shards_meta_cache > 0);
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
  return DAMACY_OK;
Invalid:
  return DAMACY_INVAL;
}

uint64_t
resolve_max_chunk_uncompressed(const struct damacy_config* cfg)
{
  uint64_t v = cfg->tuning.max_chunk_uncompressed_bytes;
  if (v == 0)
    v = DAMACY_DEFAULT_CHUNK_UNCOMPRESSED_BYTES;
  return v;
}

uint64_t
resolve_max_read_op_bytes(const struct damacy_config* cfg)
{
  uint64_t v = cfg->tuning.max_read_op_bytes;
  if (v == 0)
    v = DAMACY_DEFAULT_READ_OP_MAX_BYTES;
  return v;
}

uint8_t
resolve_host_buffer_waves(const struct damacy_config* cfg)
{
  uint8_t v = cfg->tuning.host_buffer_waves;
  if (v == 0)
    v = DAMACY_DEFAULT_HOST_BUFFER_WAVES;
  if (v < DAMACY_N_WAVES)
    v = DAMACY_N_WAVES;
  if (v > DAMACY_MAX_HOST_BUFFER_WAVES)
    v = DAMACY_MAX_HOST_BUFFER_WAVES;
  return v;
}

uint32_t
resolve_max_chunks_per_wave(const struct damacy_config* cfg)
{
  uint32_t v = cfg->tuning.max_chunks_per_wave;
  if (v == 0)
    v = DAMACY_DEFAULT_MAX_CHUNKS_PER_WAVE;
  if (v > DAMACY_HARD_MAX_CHUNKS_PER_WAVE)
    v = DAMACY_HARD_MAX_CHUNKS_PER_WAVE;
  return v;
}

uint32_t
resolve_max_substreams_per_chunk(const struct damacy_config* cfg)
{
  uint32_t v = cfg->tuning.max_substreams_per_chunk;
  if (v == 0)
    v = DAMACY_DEFAULT_MAX_SUBSTREAMS_PER_CHUNK;
  if (v > DAMACY_HARD_MAX_SUBSTREAMS_PER_CHUNK)
    v = DAMACY_HARD_MAX_SUBSTREAMS_PER_CHUNK;
  return v;
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
  uint64_t volume = 1;
  for (uint8_t d = 0; d < rank; ++d)
    volume *= (uint64_t)shape[d];
  *out_bytes =
    volume * (uint64_t)cfg->batch_size * damacy_dtype_bpe(cfg->dtype);
  return DAMACY_OK;
}
