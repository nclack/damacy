#include "damacy_config.h"

#include "damacy_limits.h"
#include "util/prelude.h"

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
  CHECK_SILENT(Invalid, cfg->host_buffer_bytes > 0);
  CHECK_SILENT(Invalid, cfg->device_buffer_bytes > 0);
  CHECK_SILENT(Invalid, cfg->n_zarrs_meta_cache > 0);
  CHECK_SILENT(Invalid, cfg->n_shards_meta_cache > 0);
  CHECK_SILENT(Invalid, damacy_dtype_bpe(cfg->dtype) > 0);
  CHECK_SILENT(Invalid,
               cfg->max_bytes_per_element <= DAMACY_BLOSC_MAX_TYPESIZE);
  CHECK_SILENT(Invalid,
               cfg->max_chunk_uncompressed_bytes <=
                 DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES);
  return DAMACY_OK;
Invalid:
  return DAMACY_INVAL;
}

uint8_t
resolve_max_bpe(const struct damacy_config* cfg)
{
  return cfg->max_bytes_per_element ? cfg->max_bytes_per_element
                                    : (uint8_t)DAMACY_BLOSC_MAX_TYPESIZE;
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
