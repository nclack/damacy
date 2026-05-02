#include "decoder.h"

#include <nvcomp/zstd.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK_CUDA(expr)                                                          \
  do {                                                                         \
    cudaError_t _e = (expr);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr,                                                          \
              "[decoder] %s:%d: %s -> %s\n",                                   \
              __FILE__,                                                        \
              __LINE__,                                                        \
              #expr,                                                           \
              cudaGetErrorString(_e));                                         \
      goto fail;                                                               \
    }                                                                          \
  } while (0)

#define CK_NVCOMP(expr)                                                        \
  do {                                                                         \
    nvcompStatus_t _s = (expr);                                                \
    if (_s != nvcompSuccess) {                                                 \
      fprintf(stderr,                                                          \
              "[decoder] %s:%d: %s -> nvcompStatus=%d\n",                      \
              __FILE__,                                                        \
              __LINE__,                                                        \
              #expr,                                                           \
              (int)_s);                                                        \
      goto fail;                                                               \
    }                                                                          \
  } while (0)

struct decoder
{
  int device_id;
  size_t max_batch;
  size_t max_chunk_uncompressed;

  // Device-side fanout buffers (sized for max_batch).
  const void** d_compressed_ptrs; // [n] device pointers to compressed buffers
  size_t* d_compressed_sizes;
  size_t* d_uncompressed_buffer_sizes;
  size_t* d_uncompressed_actual_sizes;
  void** d_uncompressed_ptrs;
  nvcompStatus_t* d_statuses;

  // Host staging for the host->device fanout copies. Pinned for async copies.
  const void** h_compressed_ptrs;
  size_t* h_compressed_sizes;
  size_t* h_uncompressed_buffer_sizes;
  void** h_uncompressed_ptrs;

  // nvcomp temp workspace.
  void* d_temp;
  size_t temp_bytes;
};

static void
decoder_free_internal(struct decoder* d)
{
  if (!d)
    return;
  cudaFree(d->d_compressed_ptrs);
  cudaFree(d->d_compressed_sizes);
  cudaFree(d->d_uncompressed_buffer_sizes);
  cudaFree(d->d_uncompressed_actual_sizes);
  cudaFree(d->d_uncompressed_ptrs);
  cudaFree(d->d_statuses);
  cudaFree(d->d_temp);
  cudaFreeHost(d->h_compressed_ptrs);
  cudaFreeHost(d->h_compressed_sizes);
  cudaFreeHost(d->h_uncompressed_buffer_sizes);
  cudaFreeHost(d->h_uncompressed_ptrs);
  free(d);
}

struct decoder*
decoder_create(int device_id,
               size_t max_batch_size,
               size_t max_chunk_uncompressed_bytes)
{
  if (max_batch_size == 0 || max_chunk_uncompressed_bytes == 0) {
    fprintf(stderr,
            "[decoder] bad args: max_batch=%zu max_chunk=%zu\n",
            max_batch_size,
            max_chunk_uncompressed_bytes);
    return NULL;
  }
  cudaError_t e0 = cudaSetDevice(device_id);
  if (e0 != cudaSuccess) {
    fprintf(stderr,
            "[decoder] cudaSetDevice(%d) -> %s\n",
            device_id,
            cudaGetErrorString(e0));
    return NULL;
  }

  struct decoder* d = (struct decoder*)calloc(1, sizeof(*d));
  if (!d)
    return NULL;
  d->device_id = device_id;
  d->max_batch = max_batch_size;
  d->max_chunk_uncompressed = max_chunk_uncompressed_bytes;

  // Device-side fanout arrays.
  CK_CUDA(
    cudaMalloc((void**)&d->d_compressed_ptrs, max_batch_size * sizeof(void*)));
  CK_CUDA(cudaMalloc((void**)&d->d_compressed_sizes,
                     max_batch_size * sizeof(size_t)));
  CK_CUDA(cudaMalloc((void**)&d->d_uncompressed_buffer_sizes,
                     max_batch_size * sizeof(size_t)));
  CK_CUDA(cudaMalloc((void**)&d->d_uncompressed_actual_sizes,
                     max_batch_size * sizeof(size_t)));
  CK_CUDA(cudaMalloc((void**)&d->d_uncompressed_ptrs,
                     max_batch_size * sizeof(void*)));
  CK_CUDA(cudaMalloc((void**)&d->d_statuses,
                     max_batch_size * sizeof(nvcompStatus_t)));

  // Pinned host staging.
  CK_CUDA(cudaMallocHost((void**)&d->h_compressed_ptrs,
                         max_batch_size * sizeof(void*)));
  CK_CUDA(cudaMallocHost((void**)&d->h_compressed_sizes,
                         max_batch_size * sizeof(size_t)));
  CK_CUDA(cudaMallocHost((void**)&d->h_uncompressed_buffer_sizes,
                         max_batch_size * sizeof(size_t)));
  CK_CUDA(cudaMallocHost((void**)&d->h_uncompressed_ptrs,
                         max_batch_size * sizeof(void*)));

  // Query temp size and allocate.
  nvcompBatchedZstdDecompressOpts_t opts =
    nvcompBatchedZstdDecompressDefaultOpts;
  size_t temp_bytes = 0;
  CK_NVCOMP(nvcompBatchedZstdDecompressGetTempSizeAsync(
    max_batch_size,
    max_chunk_uncompressed_bytes,
    opts,
    &temp_bytes,
    max_batch_size * max_chunk_uncompressed_bytes));
  d->temp_bytes = temp_bytes;
  if (temp_bytes > 0) {
    CK_CUDA(cudaMalloc(&d->d_temp, temp_bytes));
  }

  return d;

fail:
  decoder_free_internal(d);
  return NULL;
}

void
decoder_destroy(struct decoder* d)
{
  decoder_free_internal(d);
}

int
decoder_decompress_batch(struct decoder* d,
                         cudaStream_t stream,
                         const void* const* compressed,
                         const size_t* compressed_sizes,
                         void* const* decompressed,
                         const size_t* uncompressed_sizes,
                         size_t n)
{
  if (!d || n == 0 || n > d->max_batch)
    return 1;
  if (!compressed || !compressed_sizes || !decompressed || !uncompressed_sizes)
    return 1;

  // Copy host-side arrays to pinned host staging, then async H2D into the
  // device fanout arrays nvcomp will read.
  for (size_t i = 0; i < n; ++i) {
    d->h_compressed_ptrs[i] = compressed[i];
    d->h_compressed_sizes[i] = compressed_sizes[i];
    d->h_uncompressed_buffer_sizes[i] = uncompressed_sizes[i];
    d->h_uncompressed_ptrs[i] = decompressed[i];
  }
  if (cudaMemcpyAsync(d->d_compressed_ptrs,
                      d->h_compressed_ptrs,
                      n * sizeof(void*),
                      cudaMemcpyHostToDevice,
                      stream) != cudaSuccess)
    return 1;
  if (cudaMemcpyAsync(d->d_compressed_sizes,
                      d->h_compressed_sizes,
                      n * sizeof(size_t),
                      cudaMemcpyHostToDevice,
                      stream) != cudaSuccess)
    return 1;
  if (cudaMemcpyAsync(d->d_uncompressed_buffer_sizes,
                      d->h_uncompressed_buffer_sizes,
                      n * sizeof(size_t),
                      cudaMemcpyHostToDevice,
                      stream) != cudaSuccess)
    return 1;
  if (cudaMemcpyAsync(d->d_uncompressed_ptrs,
                      d->h_uncompressed_ptrs,
                      n * sizeof(void*),
                      cudaMemcpyHostToDevice,
                      stream) != cudaSuccess)
    return 1;

  nvcompBatchedZstdDecompressOpts_t opts =
    nvcompBatchedZstdDecompressDefaultOpts;
  nvcompStatus_t s =
    nvcompBatchedZstdDecompressAsync((const void* const*)d->d_compressed_ptrs,
                                     d->d_compressed_sizes,
                                     d->d_uncompressed_buffer_sizes,
                                     d->d_uncompressed_actual_sizes,
                                     n,
                                     d->d_temp,
                                     d->temp_bytes,
                                     (void* const*)d->d_uncompressed_ptrs,
                                     opts,
                                     d->d_statuses,
                                     stream);
  if (s != nvcompSuccess)
    return 1;
  return 0;
}
