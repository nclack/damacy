#include "damacy_config.h"

#include "damacy_limits.h"
#include "util/prelude.h"

#include <stdlib.h>
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
  CHECK_SILENT(Invalid, cfg->lookahead_batches >= 2);
  CHECK_SILENT(Invalid, cfg->n_io_threads > 0);
  CHECK_SILENT(Invalid, cfg->n_io_threads <= DAMACY_MAX_IO_THREADS);
  CHECK_SILENT(Invalid,
               cfg->host_buffer_waves == 0 ||
                 (cfg->host_buffer_waves >= DAMACY_N_WAVES &&
                  cfg->host_buffer_waves <= DAMACY_MAX_HOST_BUFFER_WAVES));
  CHECK_SILENT(Invalid, cfg->n_zarrs_meta_cache > 0);
  CHECK_SILENT(Invalid, cfg->n_shards_meta_cache > 0);
  CHECK_SILENT(Invalid, damacy_dtype_bpe(cfg->dtype) > 0);
  CHECK_SILENT(Invalid,
               cfg->max_chunk_uncompressed_bytes <=
                 DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES);
  // numa_strategy is a small enum; AUTO=0 so designated-init callers
  // get the default. PIN_TO requires a non-negative node; AUTO and
  // DISABLED ignore numa_node.
  CHECK_SILENT(Invalid,
               cfg->numa_strategy == DAMACY_NUMA_AUTO ||
                 cfg->numa_strategy == DAMACY_NUMA_DISABLED ||
                 cfg->numa_strategy == DAMACY_NUMA_PIN_TO);
  if (cfg->numa_strategy == DAMACY_NUMA_PIN_TO)
    CHECK_SILENT(Invalid, cfg->numa_node >= 0);
  return DAMACY_OK;
Invalid:
  return DAMACY_INVAL;
}

uint64_t
resolve_max_chunk_uncompressed(const struct damacy_config* cfg)
{
  uint64_t v = cfg->max_chunk_uncompressed_bytes;
  if (v == 0)
    v = DAMACY_DEFAULT_CHUNK_UNCOMPRESSED_BYTES;
  if (v > DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES)
    v = DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES;
  return v;
}

uint64_t
resolve_max_read_op_bytes(const struct damacy_config* cfg)
{
  uint64_t v = cfg->max_read_op_bytes;
  if (v == 0)
    v = DAMACY_DEFAULT_READ_OP_MAX_BYTES;
  return v;
}

uint64_t
resolve_max_gpu_memory(const struct damacy_config* cfg)
{
  uint64_t v = cfg->max_gpu_memory_bytes;
  if (v == 0)
    v = DAMACY_DEFAULT_MAX_GPU_MEMORY_BYTES;
  return v;
}

uint8_t
resolve_host_buffer_waves(const struct damacy_config* cfg)
{
  uint8_t v = cfg->host_buffer_waves;
  if (v == 0)
    v = DAMACY_DEFAULT_HOST_BUFFER_WAVES;
  if (v < DAMACY_N_WAVES)
    v = DAMACY_N_WAVES;
  if (v > DAMACY_MAX_HOST_BUFFER_WAVES)
    v = DAMACY_MAX_HOST_BUFFER_WAVES;
  return v;
}

uint8_t
resolve_enable_gds(const struct damacy_config* cfg)
{
  if (cfg && cfg->enable_gds)
    return 1;
  const char* e = getenv("DAMACY_GDS_ENABLE");
  if (e && strcmp(e, "1") == 0)
    return 1;
  return 0;
}
