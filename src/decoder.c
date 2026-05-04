#include "decoder.h"

#include "log/log.h"
#include "util/prelude.h"

#include <nvcomp/zstd.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CU(label, expr)                                                        \
  do {                                                                         \
    cudaError_t _e = (expr);                                                   \
    if (_e != cudaSuccess) {                                                   \
      log_error("cuda: %s -> %s", #expr, cudaGetErrorString(_e));              \
      goto label;                                                              \
    }                                                                          \
  } while (0)

#define NV(label, expr)                                                        \
  do {                                                                         \
    nvcompStatus_t _s = (expr);                                                \
    if (_s != nvcompSuccess) {                                                 \
      log_error("nvcomp: %s -> nvcompStatus=%d", #expr, (int)_s);              \
      goto label;                                                              \
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
  struct decoder* d = NULL;

  CHECK(Fail, max_batch_size > 0);
  CHECK(Fail, max_chunk_uncompressed_bytes > 0);
  CU(Fail, cudaSetDevice(device_id));

  d = (struct decoder*)calloc(1, sizeof(*d));
  CHECK(Fail, d);
  d->device_id = device_id;
  d->max_batch = max_batch_size;
  d->max_chunk_uncompressed = max_chunk_uncompressed_bytes;

  // Device-side fanout arrays.
  CU(Fail,
     cudaMalloc((void**)&d->d_compressed_ptrs, max_batch_size * sizeof(void*)));
  CU(Fail,
     cudaMalloc((void**)&d->d_compressed_sizes,
                max_batch_size * sizeof(size_t)));
  CU(Fail,
     cudaMalloc((void**)&d->d_uncompressed_buffer_sizes,
                max_batch_size * sizeof(size_t)));
  CU(Fail,
     cudaMalloc((void**)&d->d_uncompressed_actual_sizes,
                max_batch_size * sizeof(size_t)));
  CU(Fail,
     cudaMalloc((void**)&d->d_uncompressed_ptrs,
                max_batch_size * sizeof(void*)));
  CU(Fail,
     cudaMalloc((void**)&d->d_statuses,
                max_batch_size * sizeof(nvcompStatus_t)));

  // Pinned host staging.
  CU(Fail,
     cudaMallocHost((void**)&d->h_compressed_ptrs,
                    max_batch_size * sizeof(void*)));
  CU(Fail,
     cudaMallocHost((void**)&d->h_compressed_sizes,
                    max_batch_size * sizeof(size_t)));
  CU(Fail,
     cudaMallocHost((void**)&d->h_uncompressed_buffer_sizes,
                    max_batch_size * sizeof(size_t)));
  CU(Fail,
     cudaMallocHost((void**)&d->h_uncompressed_ptrs,
                    max_batch_size * sizeof(void*)));

  // Query temp size and allocate.
  nvcompBatchedZstdDecompressOpts_t opts =
    nvcompBatchedZstdDecompressDefaultOpts;
  size_t temp_bytes = 0;
  NV(Fail,
     nvcompBatchedZstdDecompressGetTempSizeAsync(
       max_batch_size,
       max_chunk_uncompressed_bytes,
       opts,
       &temp_bytes,
       max_batch_size * max_chunk_uncompressed_bytes));
  d->temp_bytes = temp_bytes;
  if (temp_bytes > 0)
    CU(Fail, cudaMalloc(&d->d_temp, temp_bytes));

  return d;

Fail:
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
  CHECK(Fail, d);
  CHECK(Fail, n > 0);
  CHECK(Fail, n <= d->max_batch);
  CHECK(Fail, compressed);
  CHECK(Fail, compressed_sizes);
  CHECK(Fail, decompressed);
  CHECK(Fail, uncompressed_sizes);

  // Copy host-side arrays to pinned host staging, then async H2D into the
  // device fanout arrays nvcomp will read.
  for (size_t i = 0; i < n; ++i) {
    d->h_compressed_ptrs[i] = compressed[i];
    d->h_compressed_sizes[i] = compressed_sizes[i];
    d->h_uncompressed_buffer_sizes[i] = uncompressed_sizes[i];
    d->h_uncompressed_ptrs[i] = decompressed[i];
  }
  CU(Fail,
     cudaMemcpyAsync(d->d_compressed_ptrs,
                     d->h_compressed_ptrs,
                     n * sizeof(void*),
                     cudaMemcpyHostToDevice,
                     stream));
  CU(Fail,
     cudaMemcpyAsync(d->d_compressed_sizes,
                     d->h_compressed_sizes,
                     n * sizeof(size_t),
                     cudaMemcpyHostToDevice,
                     stream));
  CU(Fail,
     cudaMemcpyAsync(d->d_uncompressed_buffer_sizes,
                     d->h_uncompressed_buffer_sizes,
                     n * sizeof(size_t),
                     cudaMemcpyHostToDevice,
                     stream));
  CU(Fail,
     cudaMemcpyAsync(d->d_uncompressed_ptrs,
                     d->h_uncompressed_ptrs,
                     n * sizeof(void*),
                     cudaMemcpyHostToDevice,
                     stream));

  nvcompBatchedZstdDecompressOpts_t opts =
    nvcompBatchedZstdDecompressDefaultOpts;
  NV(Fail,
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
                                      stream));
  return 0;

Fail:
  return 1;
}
