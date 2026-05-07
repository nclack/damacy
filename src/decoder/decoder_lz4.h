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
    size_t max_substream_uncompressed_bytes,
    size_t max_total_uncompressed_bytes);

  void decoder_lz4_destroy(struct decoder_lz4*);

  // Stream-async batched LZ4 decompress. The "batch" elements are
  // sub-streams: with blosc1's typesize-split, one source chunk produces
  // typesize sub-streams per block. The four input arrays are already
  // on the device — populated by an upstream GPU kernel (blosc1 emit).
  int decoder_lz4_batch_device(struct decoder_lz4*,
                               CUstream stream,
                               const void* const* d_compressed,
                               const size_t* d_compressed_sizes,
                               void* const* d_decompressed,
                               const size_t* d_uncompressed_sizes,
                               size_t n);

  // Pointer to the decoder's per-batch nvcompStatus_t array. See
  // decoder_zstd_d_statuses for rationale.
  const int* decoder_lz4_d_statuses(const struct decoder_lz4*);

  // Query nvcomp for the temp scratch size decoder_lz4_create will use
  // given the same caps. No allocation; counterpart to the zstd query.
  int decoder_lz4_query_temp_bytes(size_t max_batch_size,
                                   size_t max_substream_uncompressed_bytes,
                                   size_t max_total_uncompressed_bytes,
                                   size_t* out_bytes);

#ifdef __cplusplus
}
#endif
