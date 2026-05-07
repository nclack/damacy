#pragma once

#include <cuda.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct decoder_zstd;

  // max_chunk_uncompressed_bytes is the per-substream upper bound
  // (controls nvcomp's per-block scratch). max_total_uncompressed_bytes
  // is the upper bound on the SUM of decompressed bytes across one
  // batch — typically the per-wave decompressed-arena cap, much smaller
  // than max_batch_size * max_chunk_uncompressed_bytes when batch size
  // is dominated by many small substreams.
  struct decoder_zstd* decoder_zstd_create(size_t max_batch_size,
                                           size_t max_chunk_uncompressed_bytes,
                                           size_t max_total_uncompressed_bytes);

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

  // Same as decoder_zstd_batch but the four input arrays are already on
  // the device. Used when an upstream GPU kernel populated the fanout
  // directly (blosc1 emit). The decoder still owns the temp workspace,
  // actual-size and status output arrays.
  int decoder_zstd_batch_device(struct decoder_zstd*,
                                CUstream stream,
                                const void* const* d_compressed,
                                const size_t* d_compressed_sizes,
                                void* const* d_decompressed,
                                const size_t* d_uncompressed_sizes,
                                size_t n);

#ifdef __cplusplus
}
#endif
