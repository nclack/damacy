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

  // cuda_skip=1 skips the device frees (on context-lost teardown) but
  // still releases the host struct so it doesn't leak.
  void decoder_zstd_destroy(struct decoder_zstd*, int cuda_skip);

  // Stream-async batched zstd decompress. The four input arrays are
  // already on the device — typically populated by an upstream GPU
  // kernel (blosc1 emit). The decoder still owns the temp workspace,
  // actual-size and status output arrays.
  int decoder_zstd_batch_device(struct decoder_zstd*,
                                CUstream stream,
                                const void* const* d_compressed,
                                const size_t* d_compressed_sizes,
                                void* const* d_decompressed,
                                const size_t* d_uncompressed_sizes,
                                size_t n);

  // Pointer to the decoder's per-batch nvcompStatus_t array (sized for
  // max_batch). Returned as `const int*` to keep nvcomp out of consumer
  // headers; nvcompStatus_t fits in int. Hand to
  // decoder_status_reduce_launch after a successful batch_device call.
  const int* decoder_zstd_d_statuses(const struct decoder_zstd*);

  // Query nvcomp for the temp scratch size that decoder_zstd_create will
  // request given the same caps. No allocation, no CUDA context required.
  // Returns 0 on success and writes *out_bytes.
  int decoder_zstd_query_temp_bytes(size_t max_batch_size,
                                    size_t max_chunk_uncompressed_bytes,
                                    size_t max_total_uncompressed_bytes,
                                    size_t* out_bytes);

#ifdef __cplusplus
}
#endif
