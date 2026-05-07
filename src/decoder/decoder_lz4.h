#pragma once

#include <cuda.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct decoder_lz4;

  struct decoder_lz4* decoder_lz4_create(
    size_t max_batch_size,
    size_t max_substream_uncompressed_bytes);

  void decoder_lz4_destroy(struct decoder_lz4*);

  // Stream-async batched LZ4 decompress. The "batch" elements are
  // sub-streams: with blosc1's typesize-split, one source chunk produces
  // typesize sub-streams per block.
  int decoder_lz4_batch(struct decoder_lz4*,
                        CUstream stream,
                        const void* const* compressed,
                        const size_t* compressed_sizes,
                        void* const* decompressed,
                        const size_t* uncompressed_sizes,
                        size_t n);

#ifdef __cplusplus
}
#endif
