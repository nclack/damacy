#include "decoder.h"

#include "log/log.h"
#include "util/prelude.h"

#include <nvcomp/zstd.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// nvcomp's batched API takes `cudaStream_t`. CUstream and cudaStream_t
// are the same opaque-pointer typedef in CUDA's headers, so we can
// pass the value through unchanged. nvcomp/zstd.h pulls in
// <cuda_runtime.h> for that typedef — the only place this codebase
// touches the runtime headers. We don't call any runtime API.

#define CR(label, expr)                                                        \
  do {                                                                         \
    CUresult _r = (expr);                                                      \
    if (_r != CUDA_SUCCESS) {                                                  \
      const char* _msg = NULL;                                                 \
      cuGetErrorString(_r, &_msg);                                             \
      log_error("cu: %s -> %s", #expr, _msg ? _msg : "?");                     \
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

  // Device-side fanout buffers (sized for max_batch). Stored as the
  // typed pointers nvcomp wants; cast to/from CUdeviceptr at the
  // alloc/free/memcpy boundary via the CUDPTR helper below.
  const void** d_compressed_ptrs;
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

// Driver API uses CUdeviceptr (uint64_t) for device pointers; our
// fields are typed pointers. Cast through uintptr_t to silence
// pointer-to-int conversion diagnostics. CUdeviceptr → typed pointer
// uses the inverse cast.
#define CUDPTR(p) ((CUdeviceptr)(uintptr_t)(p))

static void
decoder_free_internal(struct decoder* d)
{
  if (!d)
    return;
  // cuMemFree on 0 returns CUDA_ERROR_INVALID_VALUE; guard each.
  void* dev_ptrs[] = {
    (void*)d->d_compressed_ptrs,
    (void*)d->d_compressed_sizes,
    (void*)d->d_uncompressed_buffer_sizes,
    (void*)d->d_uncompressed_actual_sizes,
    (void*)d->d_uncompressed_ptrs,
    (void*)d->d_statuses,
    (void*)d->d_temp,
  };
  for (size_t i = 0; i < sizeof dev_ptrs / sizeof *dev_ptrs; ++i)
    if (dev_ptrs[i])
      cuMemFree(CUDPTR(dev_ptrs[i]));
  // cuMemFreeHost on NULL is documented as a no-op.
  cuMemFreeHost(d->h_compressed_ptrs);
  cuMemFreeHost(d->h_compressed_sizes);
  cuMemFreeHost(d->h_uncompressed_buffer_sizes);
  cuMemFreeHost(d->h_uncompressed_ptrs);
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
  // Caller is expected to have a CUDA context current on this thread
  // (damacy_create retains the device's primary context). device_id is
  // tracked for logging / future reuse but we don't switch contexts here.

  d = (struct decoder*)calloc(1, sizeof(*d));
  CHECK(Fail, d);
  d->device_id = device_id;
  d->max_batch = max_batch_size;
  d->max_chunk_uncompressed = max_chunk_uncompressed_bytes;

  // Device-side fanout arrays. cuMemAlloc takes &CUdeviceptr; we
  // allocate into a local then assign through with the natural pointer
  // type. (CUDPTR() handles the inverse direction at use sites.)
  CUdeviceptr dptr = 0;
  CR(Fail, cuMemAlloc(&dptr, max_batch_size * sizeof(void*)));
  d->d_compressed_ptrs = (const void**)(uintptr_t)dptr;
  CR(Fail, cuMemAlloc(&dptr, max_batch_size * sizeof(size_t)));
  d->d_compressed_sizes = (size_t*)(uintptr_t)dptr;
  CR(Fail, cuMemAlloc(&dptr, max_batch_size * sizeof(size_t)));
  d->d_uncompressed_buffer_sizes = (size_t*)(uintptr_t)dptr;
  CR(Fail, cuMemAlloc(&dptr, max_batch_size * sizeof(size_t)));
  d->d_uncompressed_actual_sizes = (size_t*)(uintptr_t)dptr;
  CR(Fail, cuMemAlloc(&dptr, max_batch_size * sizeof(void*)));
  d->d_uncompressed_ptrs = (void**)(uintptr_t)dptr;
  CR(Fail, cuMemAlloc(&dptr, max_batch_size * sizeof(nvcompStatus_t)));
  d->d_statuses = (nvcompStatus_t*)(uintptr_t)dptr;

  // Pinned host staging.
  CR(Fail,
     cuMemAllocHost((void**)&d->h_compressed_ptrs,
                    max_batch_size * sizeof(void*)));
  CR(Fail,
     cuMemAllocHost((void**)&d->h_compressed_sizes,
                    max_batch_size * sizeof(size_t)));
  CR(Fail,
     cuMemAllocHost((void**)&d->h_uncompressed_buffer_sizes,
                    max_batch_size * sizeof(size_t)));
  CR(Fail,
     cuMemAllocHost((void**)&d->h_uncompressed_ptrs,
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
  if (temp_bytes > 0) {
    CR(Fail, cuMemAlloc(&dptr, temp_bytes));
    d->d_temp = (void*)(uintptr_t)dptr;
  }

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
                         CUstream stream,
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
  CR(Fail,
     cuMemcpyHtoDAsync(CUDPTR(d->d_compressed_ptrs),
                       d->h_compressed_ptrs,
                       n * sizeof(void*),
                       stream));
  CR(Fail,
     cuMemcpyHtoDAsync(CUDPTR(d->d_compressed_sizes),
                       d->h_compressed_sizes,
                       n * sizeof(size_t),
                       stream));
  CR(Fail,
     cuMemcpyHtoDAsync(CUDPTR(d->d_uncompressed_buffer_sizes),
                       d->h_uncompressed_buffer_sizes,
                       n * sizeof(size_t),
                       stream));
  CR(Fail,
     cuMemcpyHtoDAsync(CUDPTR(d->d_uncompressed_ptrs),
                       d->h_uncompressed_ptrs,
                       n * sizeof(void*),
                       stream));

  nvcompBatchedZstdDecompressOpts_t opts =
    nvcompBatchedZstdDecompressDefaultOpts;
  NV(Fail,
     nvcompBatchedZstdDecompressAsync(d->d_compressed_ptrs,
                                      d->d_compressed_sizes,
                                      d->d_uncompressed_buffer_sizes,
                                      d->d_uncompressed_actual_sizes,
                                      n,
                                      d->d_temp,
                                      d->temp_bytes,
                                      d->d_uncompressed_ptrs,
                                      opts,
                                      d->d_statuses,
                                      stream));
  return 0;

Fail:
  return 1;
}
