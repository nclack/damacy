#include "decoder/decoder_batch.h"

#include "decoder/decoder_check.h"
#include "util/prelude.h"

#include <nvcomp/lz4.h>
#include <nvcomp/zstd.h>

#include <stdlib.h>

_Static_assert(sizeof(nvcompStatus_t) == sizeof(int),
               "nvcompStatus_t must be int-sized for status_reduce");

struct decoder_batch
{
  size_t max_batch;
  size_t* d_uncompressed_actual_sizes;
  nvcompStatus_t* d_statuses;
  void* d_temp;
  size_t temp_bytes;
};

static nvcompBatchedLZ4DecompressOpts_t
lz4_options(void)
{
  nvcompBatchedLZ4DecompressOpts_t options =
    nvcompBatchedLZ4DecompressDefaultOpts;
  // Blosc payloads are unaligned, and corrupt input needs CUDA error reporting.
  options.backend = NVCOMP_DECOMPRESS_BACKEND_CUDA;
  return options;
}

static void
release_device_buffers(struct decoder_batch* d)
{
  if (d->d_uncompressed_actual_sizes) {
    cuMemFree(CUDPTR(d->d_uncompressed_actual_sizes));
    d->d_uncompressed_actual_sizes = NULL;
  }
  if (d->d_statuses) {
    cuMemFree(CUDPTR(d->d_statuses));
    d->d_statuses = NULL;
  }
  if (d->d_temp) {
    cuMemFree(CUDPTR(d->d_temp));
    d->d_temp = NULL;
  }
  d->temp_bytes = 0;
  d->max_batch = 0;
}

static int
alloc_device_buffers(struct decoder_batch* d,
                     size_t max_batch_size,
                     size_t max_chunk_uncompressed_bytes,
                     size_t max_total_uncompressed_bytes)
{
  CHECK(Fail, max_batch_size > 0);
  CHECK(Fail, max_chunk_uncompressed_bytes > 0);

  CUdeviceptr dptr = 0;
  CU(Fail, cuMemAlloc(&dptr, max_batch_size * sizeof(size_t)));
  d->d_uncompressed_actual_sizes = (size_t*)(uintptr_t)dptr;
  CU(Fail, cuMemAlloc(&dptr, max_batch_size * sizeof(nvcompStatus_t)));
  d->d_statuses = (nvcompStatus_t*)(uintptr_t)dptr;

  size_t temp_bytes = 0;
  CHECK(Fail,
        decoder_batch_query_temp_bytes(max_batch_size,
                                       max_chunk_uncompressed_bytes,
                                       max_total_uncompressed_bytes,
                                       &temp_bytes) == 0);
  d->temp_bytes = temp_bytes;
  if (temp_bytes > 0) {
    CU(Fail, cuMemAlloc(&dptr, temp_bytes));
    d->d_temp = (void*)(uintptr_t)dptr;
  }
  d->max_batch = max_batch_size;
  return 0;
Fail:
  return 1;
}

void
decoder_batch_destroy(struct decoder_batch* d, int cuda_skip)
{
  if (!d)
    return;
  if (!cuda_skip)
    release_device_buffers(d);
  free(d);
}

struct decoder_batch*
decoder_batch_create(size_t max_batch_size,
                     size_t max_chunk_uncompressed_bytes,
                     size_t max_total_uncompressed_bytes)
{
  struct decoder_batch* d = (struct decoder_batch*)calloc(1, sizeof(*d));
  CHECK(Fail, d);
  if (alloc_device_buffers(d,
                           max_batch_size,
                           max_chunk_uncompressed_bytes,
                           max_total_uncompressed_bytes) != 0)
    goto Fail;
  return d;
Fail:
  decoder_batch_destroy(d, 0);
  return NULL;
}

size_t
decoder_batch_cur_max_batch(const struct decoder_batch* d)
{
  return d ? d->max_batch : 0;
}

int
decoder_batch_grow(struct decoder_batch* d,
                   size_t new_max_batch_size,
                   size_t max_chunk_uncompressed_bytes,
                   size_t max_total_uncompressed_bytes)
{
  CHECK(Fail, d);
  CHECK(Fail, d->max_batch && new_max_batch_size >= d->max_batch);
  release_device_buffers(d);
  if (alloc_device_buffers(d,
                           new_max_batch_size,
                           max_chunk_uncompressed_bytes,
                           max_total_uncompressed_bytes) != 0)
    goto Fail;
  return 0;
Fail:
  return 1;
}

const int*
decoder_batch_d_statuses(const struct decoder_batch* d)
{
  return d ? (const int*)d->d_statuses : NULL;
}

const size_t*
decoder_batch_d_actual_sizes(const struct decoder_batch* d)
{
  return d ? d->d_uncompressed_actual_sizes : NULL;
}

int
decoder_batch_query_temp_bytes(size_t max_batch_size,
                               size_t max_block_bytes,
                               size_t max_total_bytes,
                               size_t* out_bytes)
{
  CHECK(Fail,
        out_bytes && max_batch_size && max_block_bytes && max_total_bytes);
  size_t zstd_bytes = 0, lz4_bytes = 0;
  NV(Fail,
     nvcompBatchedZstdDecompressGetTempSizeAsync(
       max_batch_size,
       max_block_bytes,
       nvcompBatchedZstdDecompressDefaultOpts,
       &zstd_bytes,
       max_total_bytes));
  NV(Fail,
     nvcompBatchedLZ4DecompressGetTempSizeAsync(max_batch_size,
                                                max_block_bytes,
                                                lz4_options(),
                                                &lz4_bytes,
                                                max_total_bytes));
  *out_bytes = zstd_bytes > lz4_bytes ? zstd_bytes : lz4_bytes;
  return 0;
Fail:
  return 1;
}

int
decoder_batch_device(struct decoder_batch* d,
                     CUstream stream,
                     enum decoder_codec codec,
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

  if (codec == DECODER_ZSTD) {
    NV(Fail,
       nvcompBatchedZstdDecompressAsync(d_compressed,
                                        d_compressed_sizes,
                                        d_uncompressed_sizes,
                                        d->d_uncompressed_actual_sizes,
                                        n,
                                        d->d_temp,
                                        d->temp_bytes,
                                        d_decompressed,
                                        nvcompBatchedZstdDecompressDefaultOpts,
                                        d->d_statuses,
                                        stream));
  } else if (codec == DECODER_LZ4) {
    NV(Fail,
       nvcompBatchedLZ4DecompressAsync(d_compressed,
                                       d_compressed_sizes,
                                       d_uncompressed_sizes,
                                       d->d_uncompressed_actual_sizes,
                                       n,
                                       d->d_temp,
                                       d->temp_bytes,
                                       d_decompressed,
                                       lz4_options(),
                                       d->d_statuses,
                                       stream));
  } else {
    goto Fail;
  }
  return 0;

Fail:
  return 1;
}
