#pragma once

#include <cuda.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct decoder_zstd;

  struct decoder_zstd* decoder_zstd_create(size_t max_batch_size,
                                           size_t max_chunk_uncompressed_bytes);

  void decoder_zstd_destroy(struct decoder_zstd*);

  // Stream-async batched zstd decompress. All pointers are device pointers;
  // host-side arrays are staged through internal pinned host buffers.
  int decoder_zstd_batch(struct decoder_zstd*,
                         CUstream stream,
                         const void* const* compressed,
                         const size_t* compressed_sizes,
                         void* const* decompressed,
                         const size_t* uncompressed_sizes,
                         size_t n);

#ifdef __cplusplus
}
#endif
