#pragma once

#include <cuda.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  enum decoder_codec
  {
    DECODER_ZSTD,
    DECODER_LZ4,
  };

  struct decoder_batch;

  struct decoder_batch* decoder_batch_create(size_t max_batch_size,
                                             size_t max_block_bytes,
                                             size_t max_total_bytes);
  void decoder_batch_destroy(struct decoder_batch*, int cuda_skip);
  size_t decoder_batch_cur_max_batch(const struct decoder_batch*);

  // The shared workspace must be idle before growing and used on one stream.
  int decoder_batch_grow(struct decoder_batch*,
                         size_t max_batch_size,
                         size_t max_block_bytes,
                         size_t max_total_bytes);
  int decoder_batch_device(struct decoder_batch*,
                           CUstream stream,
                           enum decoder_codec codec,
                           const void* const* d_compressed,
                           const size_t* d_compressed_sizes,
                           void* const* d_decompressed,
                           const size_t* d_uncompressed_sizes,
                           size_t n);
  const int* decoder_batch_d_statuses(const struct decoder_batch*);
  const size_t* decoder_batch_d_actual_sizes(const struct decoder_batch*);
  int decoder_batch_query_temp_bytes(size_t max_batch_size,
                                     size_t max_block_bytes,
                                     size_t max_total_bytes,
                                     size_t* out_bytes);

#ifdef __cplusplus
}
#endif
