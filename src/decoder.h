// nvcomp batched zstd decompress shim. C-callable.
//
// Owns the per-batch scratch needed by nvcomp's batched API:
//   - device-side arrays of compressed/decompressed pointers and sizes
//   - the temp workspace recommended by nvcomp
//   - per-chunk status array
//
// All buffers passed to decoder_decompress_batch are *device* pointers. The
// caller is responsible for getting compressed bytes onto the device first
// (e.g., via cudaMemcpyAsync from a pinned host staging ring).
#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct decoder;

  // device_id selects the CUDA device on which scratch buffers live.
  // max_batch_size and max_chunk_uncompressed_bytes are upper bounds the
  // pipeline guarantees per call. Returns NULL on failure.
  struct decoder* decoder_create(int device_id,
                                 size_t max_batch_size,
                                 size_t max_chunk_uncompressed_bytes);

  void decoder_destroy(struct decoder* d);

  // Stream-async batched zstd decompress.
  //
  // Inputs (host-side arrays):
  //   compressed[i]          device pointer to compressed bytes for chunk i
  //   compressed_sizes[i]    compressed length for chunk i
  //   uncompressed_sizes[i]  output buffer size for chunk i (== chunk volume ×
  //   bpe) decompressed[i]        device pointer to output buffer for chunk i
  //
  // Returns 0 on successful submission. Decompression itself is async on
  // `stream`; the caller is expected to record an event after the call to
  // sequence dependent work.
  int decoder_decompress_batch(struct decoder* d,
                               cudaStream_t stream,
                               const void* const* compressed,
                               const size_t* compressed_sizes,
                               void* const* decompressed,
                               const size_t* uncompressed_sizes,
                               size_t n);

#ifdef __cplusplus
}
#endif
