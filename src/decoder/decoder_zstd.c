#include "decoder/decoder_zstd.h"

#include "decoder/decoder_check.h"
#include "util/prelude.h"

#include <nvcomp/zstd.h>

#include <stdlib.h>

// nvcompStatus_t is reported via decoder_zstd_d_statuses as `const int*`;
// guard against an ABI surprise.
_Static_assert(sizeof(nvcompStatus_t) == sizeof(int),
               "nvcompStatus_t must be int-sized for status_reduce");

// Per-batch nvcomp output arrays (actual sizes, statuses) and the temp
// workspace are owned here. The four input arrays come from the caller's
// device-side fanout (populated by the blosc1 emit kernel). max_batch is
// the *current* allocated cap and grows in lockstep with the pool's
// fanout SOA when a wave's substream count exceeds it.
struct decoder_zstd
{
  size_t max_batch;
  size_t* d_uncompressed_actual_sizes;
  nvcompStatus_t* d_statuses;
  void* d_temp;
  size_t temp_bytes;
};

// Frees the three device buffers; called by destroy and by grow before
// re-alloc. Sets the freed slots back to NULL so a subsequent grow
// failure leaves the struct safely re-destroyable.
static void
release_device_buffers(struct decoder_zstd* d)
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

// Allocates the three device buffers for the given cap. Caller must have
// freed any prior buffers first. Returns 0 on success; on failure d is
// in the partial state release_device_buffers expects (every set slot is
// safe to free), and 1 is returned.
static int
alloc_device_buffers(struct decoder_zstd* d,
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

  nvcompBatchedZstdDecompressOpts_t opts =
    nvcompBatchedZstdDecompressDefaultOpts;
  size_t temp_bytes = 0;
  NV(Fail,
     nvcompBatchedZstdDecompressGetTempSizeAsync(max_batch_size,
                                                 max_chunk_uncompressed_bytes,
                                                 opts,
                                                 &temp_bytes,
                                                 max_total_uncompressed_bytes));
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
decoder_zstd_destroy(struct decoder_zstd* d, int cuda_skip)
{
  if (!d)
    return;
  if (!cuda_skip)
    release_device_buffers(d);
  free(d);
}

struct decoder_zstd*
decoder_zstd_create(size_t max_batch_size,
                    size_t max_chunk_uncompressed_bytes,
                    size_t max_total_uncompressed_bytes)
{
  struct decoder_zstd* d = (struct decoder_zstd*)calloc(1, sizeof(*d));
  CHECK(Fail, d);
  if (alloc_device_buffers(d,
                           max_batch_size,
                           max_chunk_uncompressed_bytes,
                           max_total_uncompressed_bytes) != 0)
    goto Fail;
  return d;
Fail:
  decoder_zstd_destroy(d, 0);
  return NULL;
}

size_t
decoder_zstd_cur_max_batch(const struct decoder_zstd* d)
{
  return d ? d->max_batch : 0;
}

int
decoder_zstd_grow(struct decoder_zstd* d,
                  size_t new_max_batch_size,
                  size_t max_chunk_uncompressed_bytes,
                  size_t max_total_uncompressed_bytes)
{
  CHECK(Fail, d);
  CHECK(Fail, new_max_batch_size >= d->max_batch);
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
decoder_zstd_d_statuses(const struct decoder_zstd* d)
{
  // nvcompStatus_t is an enum; layout-compatible with int across the
  // platforms nvcomp ships for. Static-asserting elsewhere avoids
  // dragging nvcomp into the public header.
  return d ? (const int*)d->d_statuses : NULL;
}

int
decoder_zstd_query_temp_bytes(size_t max_batch_size,
                              size_t max_chunk_uncompressed_bytes,
                              size_t max_total_uncompressed_bytes,
                              size_t* out_bytes)
{
  CHECK(Fail, out_bytes);
  CHECK(Fail, max_batch_size > 0);
  CHECK(Fail, max_chunk_uncompressed_bytes > 0);
  *out_bytes = 0;
  nvcompBatchedZstdDecompressOpts_t opts =
    nvcompBatchedZstdDecompressDefaultOpts;
  NV(Fail,
     nvcompBatchedZstdDecompressGetTempSizeAsync(max_batch_size,
                                                 max_chunk_uncompressed_bytes,
                                                 opts,
                                                 out_bytes,
                                                 max_total_uncompressed_bytes));
  return 0;
Fail:
  return 1;
}

int
decoder_zstd_batch_device(struct decoder_zstd* d,
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

  nvcompBatchedZstdDecompressOpts_t opts =
    nvcompBatchedZstdDecompressDefaultOpts;
  NV(Fail,
     nvcompBatchedZstdDecompressAsync(d_compressed,
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
