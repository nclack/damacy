#include "decoder/decoder_lz4.h"

#include "decoder/decoder_check.h"
#include "log/log.h"
#include "util/prelude.h"

#include <nvcomp/lz4.h>

#include <stdlib.h>
#include <string.h>

struct decoder_lz4
{
  size_t max_batch;
  size_t max_substream_uncompressed;

  const void** d_compressed_ptrs;
  size_t* d_compressed_sizes;
  size_t* d_uncompressed_buffer_sizes;
  size_t* d_uncompressed_actual_sizes;
  void** d_uncompressed_ptrs;
  nvcompStatus_t* d_statuses;

  const void** h_compressed_ptrs;
  size_t* h_compressed_sizes;
  size_t* h_uncompressed_buffer_sizes;
  void** h_uncompressed_ptrs;

  void* d_temp;
  size_t temp_bytes;
};

void
decoder_lz4_destroy(struct decoder_lz4* d)
{
  if (!d)
    return;
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
  cuMemFreeHost(d->h_compressed_ptrs);
  cuMemFreeHost(d->h_compressed_sizes);
  cuMemFreeHost(d->h_uncompressed_buffer_sizes);
  cuMemFreeHost(d->h_uncompressed_ptrs);
  free(d);
}

struct decoder_lz4*
decoder_lz4_create(size_t max_batch_size,
                   size_t max_substream_uncompressed_bytes)
{
  struct decoder_lz4* d = NULL;

  CHECK(Fail, max_batch_size > 0);
  CHECK(Fail, max_substream_uncompressed_bytes > 0);

  d = (struct decoder_lz4*)calloc(1, sizeof(*d));
  CHECK(Fail, d);
  d->max_batch = max_batch_size;
  d->max_substream_uncompressed = max_substream_uncompressed_bytes;

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

  nvcompBatchedLZ4DecompressOpts_t opts = nvcompBatchedLZ4DecompressDefaultOpts;
  size_t temp_bytes = 0;
  NV(Fail,
     nvcompBatchedLZ4DecompressGetTempSizeAsync(
       max_batch_size,
       max_substream_uncompressed_bytes,
       opts,
       &temp_bytes,
       max_batch_size * max_substream_uncompressed_bytes));
  d->temp_bytes = temp_bytes;
  if (temp_bytes > 0) {
    CR(Fail, cuMemAlloc(&dptr, temp_bytes));
    d->d_temp = (void*)(uintptr_t)dptr;
  }

  return d;

Fail:
  decoder_lz4_destroy(d);
  return NULL;
}

int
decoder_lz4_batch(struct decoder_lz4* d,
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

  nvcompBatchedLZ4DecompressOpts_t opts = nvcompBatchedLZ4DecompressDefaultOpts;
  NV(Fail,
     nvcompBatchedLZ4DecompressAsync(d->d_compressed_ptrs,
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
