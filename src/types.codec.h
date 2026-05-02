#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  enum compression_codec
  {
    CODEC_NONE,
    CODEC_LZ4_NON_STANDARD, // raw LZ4 block (not standard zarr v3)
    CODEC_ZSTD,
    CODEC_BLOSC_LZ4,
    CODEC_BLOSC_ZSTD,
  };

  enum codec_shuffle
  {
    CODEC_SHUFFLE_NONE = 0,
    CODEC_SHUFFLE_BYTE = 1,
    CODEC_SHUFFLE_BIT = 2,
  };

  struct codec_config
  {
    enum compression_codec id;
    uint8_t level;
    enum codec_shuffle shuffle;
  };

  int codec_is_blosc(enum compression_codec c);
  int codec_is_gpu_supported(enum compression_codec c);

#ifdef __cplusplus
}
#endif
