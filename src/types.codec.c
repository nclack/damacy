#include "types.codec.h"

int
codec_is_blosc(enum compression_codec c)
{
  return c == CODEC_BLOSC_LZ4 || c == CODEC_BLOSC_ZSTD;
}

int
codec_is_gpu_supported(enum compression_codec c)
{
  return c == CODEC_NONE || c == CODEC_LZ4_NON_STANDARD || c == CODEC_ZSTD;
}
