#include "decoder/decoder_lz4.h"

#include "decoder/decoder_check.h"
#include "util/prelude.h"

#include <nvcomp/lz4.h>

#include <stdlib.h>

// Per-batch nvcomp output arrays (actual sizes, statuses) and the temp
// workspace are owned here. The four input arrays come from the caller's
// device-side fanout (populated by the blosc1 emit kernel).
struct decoder_lz4
{
  size_t max_batch;
  size_t* d_uncompressed_actual_sizes;
  nvcompStatus_t* d_statuses;
  void* d_temp;
  size_t temp_bytes;
};

void
decoder_lz4_destroy(struct decoder_lz4* d)
{
  if (!d)
    return;
  void* dev_ptrs[] = {
    (void*)d->d_uncompressed_actual_sizes,
    (void*)d->d_statuses,
    (void*)d->d_temp,
  };
  for (size_t i = 0; i < sizeof dev_ptrs / sizeof *dev_ptrs; ++i)
    if (dev_ptrs[i])
      cuMemFree(CUDPTR(dev_ptrs[i]));
  free(d);
}

struct decoder_lz4*
decoder_lz4_create(size_t max_batch_size,
                   size_t max_substream_uncompressed_bytes,
                   size_t max_total_uncompressed_bytes)
{
  struct decoder_lz4* d = NULL;

  CHECK(Fail, max_batch_size > 0);
  CHECK(Fail, max_substream_uncompressed_bytes > 0);

  d = (struct decoder_lz4*)calloc(1, sizeof(*d));
  CHECK(Fail, d);
  d->max_batch = max_batch_size;

  CUdeviceptr dptr = 0;
  CR(Fail, cuMemAlloc(&dptr, max_batch_size * sizeof(size_t)));
  d->d_uncompressed_actual_sizes = (size_t*)(uintptr_t)dptr;
  CR(Fail, cuMemAlloc(&dptr, max_batch_size * sizeof(nvcompStatus_t)));
  d->d_statuses = (nvcompStatus_t*)(uintptr_t)dptr;

  nvcompBatchedLZ4DecompressOpts_t opts = nvcompBatchedLZ4DecompressDefaultOpts;
  size_t temp_bytes = 0;
  NV(
    Fail,
    nvcompBatchedLZ4DecompressGetTempSizeAsync(max_batch_size,
                                               max_substream_uncompressed_bytes,
                                               opts,
                                               &temp_bytes,
                                               max_total_uncompressed_bytes));
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
decoder_lz4_batch_device(struct decoder_lz4* d,
                         CUstream stream,
                         const void* const* d_compressed,
                         const size_t* d_compressed_sizes,
                         void* const* d_decompressed,
                         const size_t* d_uncompressed_sizes,
                         size_t n)
{
  CHECK(Fail, d);
  CHECK(Fail, n > 0);
  CHECK(Fail, n <= d->max_batch);
  CHECK(Fail, d_compressed);
  CHECK(Fail, d_compressed_sizes);
  CHECK(Fail, d_decompressed);
  CHECK(Fail, d_uncompressed_sizes);

  nvcompBatchedLZ4DecompressOpts_t opts = nvcompBatchedLZ4DecompressDefaultOpts;
  NV(Fail,
     nvcompBatchedLZ4DecompressAsync(d_compressed,
                                     d_compressed_sizes,
                                     d_uncompressed_sizes,
                                     d->d_uncompressed_actual_sizes,
                                     n,
                                     d->d_temp,
                                     d->temp_bytes,
                                     d_decompressed,
                                     opts,
                                     d->d_statuses,
                                     stream));
  return 0;

Fail:
  return 1;
}
