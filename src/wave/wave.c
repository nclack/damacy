#include "wave.h"

#include "decoder/decoder_check.h" // CU + CUDPTR
#include "log/log.h"
#include "util/prelude.h"

#include <stdlib.h>
#include <string.h>

int
fanout_alloc_pinned(struct blosc1_host_fanout* h,
                    struct nvcomp_fanout* d,
                    size_t n)
{
  CUdeviceptr dptr = 0;
  CU(Fail, cuMemAllocHost((void**)&h->comp_ptrs, n * sizeof(void*)));
  CU(Fail, cuMemAllocHost((void**)&h->comp_sizes, n * sizeof(size_t)));
  CU(Fail, cuMemAllocHost((void**)&h->decomp_ptrs, n * sizeof(void*)));
  CU(Fail, cuMemAllocHost((void**)&h->decomp_buf_sizes, n * sizeof(size_t)));
  CU(Fail, cuMemAlloc(&dptr, n * sizeof(void*)));
  d->d_comp_ptrs = (const void**)(uintptr_t)dptr;
  CU(Fail, cuMemAlloc(&dptr, n * sizeof(size_t)));
  d->d_comp_sizes = (size_t*)(uintptr_t)dptr;
  CU(Fail, cuMemAlloc(&dptr, n * sizeof(void*)));
  d->d_decomp_ptrs = (void**)(uintptr_t)dptr;
  CU(Fail, cuMemAlloc(&dptr, n * sizeof(size_t)));
  d->d_decomp_buf_sizes = (size_t*)(uintptr_t)dptr;
  return 0;
Fail:
  return 1;
}

enum damacy_status
fanout_upload(CUstream s,
              const struct nvcomp_fanout* d,
              const struct blosc1_host_fanout* h,
              size_t n)
{
  CU(Fail,
     cuMemcpyHtoDAsync(
       CUDPTR(d->d_comp_ptrs), h->comp_ptrs, n * sizeof(void*), s));
  CU(Fail,
     cuMemcpyHtoDAsync(
       CUDPTR(d->d_comp_sizes), h->comp_sizes, n * sizeof(size_t), s));
  CU(Fail,
     cuMemcpyHtoDAsync(
       CUDPTR(d->d_decomp_ptrs), h->decomp_ptrs, n * sizeof(void*), s));
  CU(Fail,
     cuMemcpyHtoDAsync(CUDPTR(d->d_decomp_buf_sizes),
                       h->decomp_buf_sizes,
                       n * sizeof(size_t),
                       s));
  return DAMACY_OK;
Fail:
  return DAMACY_CUDA;
}

int
wave_init(struct damacy_wave* wave,
          uint64_t host_slab_bytes,
          uint64_t dev_decompressed_bytes,
          uint8_t max_bpe,
          uint64_t max_chunk_uncompressed_bytes)
{
  wave->state = WAVE_FREE;
  wave->host_slab_cap = host_slab_bytes;
  wave->dev_decompressed_cap = dev_decompressed_bytes;

  CU(Error, cuMemAllocHost(&wave->host_slab, host_slab_bytes));
  CUdeviceptr dptr = 0;
  CU(Error, cuMemAlloc(&dptr, host_slab_bytes));
  wave->dev_compressed = (void*)(uintptr_t)dptr;
  CU(Error, cuMemAlloc(&dptr, dev_decompressed_bytes));
  wave->dev_decompressed = (void*)(uintptr_t)dptr;

  uint32_t cap = DAMACY_MAX_CHUNKS_PER_WAVE;
  CU(Error,
     cuMemAllocHost((void**)&wave->h_chunks,
                    (size_t)cap * sizeof(struct blosc1_host_chunk)));
  CU(Error,
     cuMemAllocHost((void**)&wave->scratch.hdrs,
                    (size_t)cap * sizeof(struct blosc1_chunk_hdr)));
  CU(Error,
     cuMemAllocHost((void**)&wave->scratch.counts,
                    (size_t)cap * sizeof(struct blosc1_chunk_counts)));
  CU(Error,
     cuMemAllocHost((void**)&wave->scratch.offsets,
                    (size_t)cap * sizeof(struct blosc1_chunk_offsets)));
  // Pure host scratch, but pinned for consistency with the rest of the
  // scratch arrays. cap * MAX_BLOCKS uint32_t each (~64 KB at current caps).
  CU(Error,
     cuMemAllocHost((void**)&wave->scratch.bstarts,
                    (size_t)cap * DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK *
                      sizeof(uint32_t)));
  CU(Error,
     cuMemAllocHost((void**)&wave->scratch.block_ends,
                    (size_t)cap * DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK *
                      sizeof(uint32_t)));
  CU(Error,
     cuMemAllocHost((void**)&wave->h_blosc1_totals,
                    sizeof(struct blosc1_totals)));
  wave->h_assemble_chunks =
    (struct assemble_chunk*)calloc(cap, sizeof(struct assemble_chunk));
  CHECK(Error, wave->h_assemble_chunks);
  wave->store_reads =
    (struct store_read*)calloc(cap, sizeof(struct store_read));
  CHECK(Error, wave->store_reads);

  CU(Error, cuMemAlloc(&dptr, (size_t)cap * sizeof(struct assemble_chunk)));
  wave->d_assemble_chunks = (struct assemble_chunk*)(uintptr_t)dptr;

  CU(Error, cuMemAlloc(&dptr, sizeof(struct blosc1_totals)));
  wave->d_blosc1_totals = (struct blosc1_totals*)(uintptr_t)dptr;

  const size_t zsubs = DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE;
  const size_t lsubs = lz4_subs_per_wave(max_bpe);
  if (fanout_alloc_pinned(&wave->h_zstd_fan, &wave->zstd_fan, zsubs))
    goto Error;
  if (fanout_alloc_pinned(&wave->h_lz4_fan, &wave->lz4_fan, lsubs))
    goto Error;

  CU(Error,
     cuMemAllocHost((void**)&wave->h_memcpy_ops,
                    DAMACY_MAX_BLOSC_MEMCPY_OPS_PER_WAVE *
                      sizeof(struct gpu_memcpy_op)));
  CU(Error,
     cuMemAllocHost((void**)&wave->h_unshuffle_ops,
                    DAMACY_MAX_BLOSC_SHUFFLE_OPS_PER_WAVE *
                      sizeof(struct gpu_shuffle_op)));
  CU(Error,
     cuMemAllocHost((void**)&wave->h_bitunshuffle_ops,
                    DAMACY_MAX_BLOSC_SHUFFLE_OPS_PER_WAVE *
                      sizeof(struct gpu_shuffle_op)));
  CU(Error,
     cuMemAlloc(&dptr,
                DAMACY_MAX_BLOSC_MEMCPY_OPS_PER_WAVE *
                  sizeof(struct gpu_memcpy_op)));
  wave->d_memcpy_ops = (struct gpu_memcpy_op*)(uintptr_t)dptr;
  CU(Error,
     cuMemAlloc(&dptr,
                DAMACY_MAX_BLOSC_SHUFFLE_OPS_PER_WAVE *
                  sizeof(struct gpu_shuffle_op)));
  wave->d_unshuffle_ops = (struct gpu_shuffle_op*)(uintptr_t)dptr;
  CU(Error,
     cuMemAlloc(&dptr,
                DAMACY_MAX_BLOSC_SHUFFLE_OPS_PER_WAVE *
                  sizeof(struct gpu_shuffle_op)));
  wave->d_bitunshuffle_ops = (struct gpu_shuffle_op*)(uintptr_t)dptr;
  CU(Error, cuMemAlloc(&dptr, dev_decompressed_bytes));
  wave->dev_unshuffle_scratch = (void*)(uintptr_t)dptr;

  CU(Error, cuEventCreate(&wave->ev.h2d_start, CU_EVENT_DEFAULT));
  CU(Error, cuEventCreate(&wave->ev.bulk_h2d_end, CU_EVENT_DEFAULT));
  CU(Error, cuEventCreate(&wave->ev.h2d_end, CU_EVENT_DEFAULT));
  CU(Error, cuEventCreate(&wave->ev.decomp_start, CU_EVENT_DEFAULT));
  CU(Error, cuEventCreate(&wave->ev.zstd_done, CU_EVENT_DEFAULT));
  CU(Error, cuEventCreate(&wave->ev.lz4_done, CU_EVENT_DEFAULT));
  CU(Error, cuEventCreate(&wave->ev.post_start, CU_EVENT_DEFAULT));
  CU(Error, cuEventCreate(&wave->ev.decomp_end, CU_EVENT_DEFAULT));
  CU(Error, cuEventCreate(&wave->ev.asm_start, CU_EVENT_DEFAULT));
  CU(Error, cuEventCreate(&wave->ev.asm_end, CU_EVENT_DEFAULT));

  // nvcomp temp scratch is sized off min(runtime per-chunk cap × wave
  // chunks, runtime per-wave decompress budget). The runtime cap (set
  // by cfg.max_chunk_uncompressed_bytes; default 512 KB) is the lever
  // that lets users keep nvcomp scratch small on tight GPU budgets while
  // still letting the compile-time ceiling stretch to 2 MB on bigger devices.
  const size_t cap_chunks = DAMACY_MAX_CHUNKS_PER_WAVE;
  const size_t runtime_chunk_cap = (size_t)max_chunk_uncompressed_bytes;
  const size_t cap_worst = cap_chunks * runtime_chunk_cap;
  const size_t wave_total_uncompressed =
    dev_decompressed_bytes < cap_worst ? dev_decompressed_bytes : cap_worst;
  // Zstd substream == one blosc-block, ≤ chunk_uncompressed_cap.
  // LZ4 substream == blosc-block / typesize, so the per-substream cap
  // tightens by max_bpe — directly shrinks nvcomp's LZ4 temp scratch.
  size_t zstd_per_substream_cap = runtime_chunk_cap;
  if (zstd_per_substream_cap > wave_total_uncompressed &&
      wave_total_uncompressed > 0)
    zstd_per_substream_cap = wave_total_uncompressed;
  size_t lz4_per_substream_cap = zstd_per_substream_cap / (size_t)max_bpe;
  if (lz4_per_substream_cap == 0)
    lz4_per_substream_cap = 1;
  wave->zstd_decoder = decoder_zstd_create(DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE,
                                           zstd_per_substream_cap,
                                           wave_total_uncompressed);
  CHECK(Error, wave->zstd_decoder);
  wave->lz4_decoder =
    decoder_lz4_create(lsubs, lz4_per_substream_cap, wave_total_uncompressed);
  CHECK(Error, wave->lz4_decoder);
  return 0;
Error:
  // Make the partial-init cleanup explicit instead of relying on the
  // outer destroy_inner walking a zero-initialized wave.
  wave_destroy(wave, 0);
  return 1;
}

void
wave_destroy(struct damacy_wave* wave, int cuda_skip)
{
  if (!wave)
    return;
  if (!cuda_skip) {
    decoder_zstd_destroy(wave->zstd_decoder);
    decoder_lz4_destroy(wave->lz4_decoder);
    void* const host_ptrs[] = {
      wave->host_slab,
      wave->h_chunks,
      wave->scratch.hdrs,
      wave->scratch.counts,
      wave->scratch.offsets,
      wave->scratch.bstarts,
      wave->scratch.block_ends,
      wave->h_blosc1_totals,
      (void*)wave->h_zstd_fan.comp_ptrs,
      wave->h_zstd_fan.comp_sizes,
      wave->h_zstd_fan.decomp_ptrs,
      wave->h_zstd_fan.decomp_buf_sizes,
      (void*)wave->h_lz4_fan.comp_ptrs,
      wave->h_lz4_fan.comp_sizes,
      wave->h_lz4_fan.decomp_ptrs,
      wave->h_lz4_fan.decomp_buf_sizes,
      wave->h_memcpy_ops,
      wave->h_unshuffle_ops,
      wave->h_bitunshuffle_ops,
    };
    for (size_t i = 0; i < countof(host_ptrs); ++i)
      if (host_ptrs[i])
        cuMemFreeHost(host_ptrs[i]);
    void* const dev_ptrs[] = {
      wave->dev_compressed,
      wave->dev_decompressed,
      wave->d_assemble_chunks,
      wave->d_blosc1_totals,
      (void*)wave->zstd_fan.d_comp_ptrs,
      wave->zstd_fan.d_comp_sizes,
      wave->zstd_fan.d_decomp_ptrs,
      wave->zstd_fan.d_decomp_buf_sizes,
      (void*)wave->lz4_fan.d_comp_ptrs,
      wave->lz4_fan.d_comp_sizes,
      wave->lz4_fan.d_decomp_ptrs,
      wave->lz4_fan.d_decomp_buf_sizes,
      wave->d_memcpy_ops,
      wave->d_unshuffle_ops,
      wave->d_bitunshuffle_ops,
      wave->dev_unshuffle_scratch,
    };
    for (size_t i = 0; i < countof(dev_ptrs); ++i)
      if (dev_ptrs[i])
        cuMemFree(CUDPTR(dev_ptrs[i]));
    CUevent* const events[] = { &wave->ev.h2d_start,  &wave->ev.bulk_h2d_end,
                                &wave->ev.h2d_end,    &wave->ev.decomp_start,
                                &wave->ev.zstd_done,  &wave->ev.lz4_done,
                                &wave->ev.post_start, &wave->ev.decomp_end,
                                &wave->ev.asm_start,  &wave->ev.asm_end };
    for (size_t i = 0; i < countof(events); ++i)
      if (*events[i])
        cuEventDestroy_v2(*events[i]);
  }
  free(wave->h_assemble_chunks);
  free(wave->store_reads);
  memset(wave, 0, sizeof(*wave));
}

int
find_free_wave(const struct damacy_wave waves[2])
{
  for (int w = 0; w < 2; ++w)
    if (waves[w].state == WAVE_FREE)
      return w;
  return -1;
}

int
any_wave_in_flight(const struct damacy_wave waves[2])
{
  for (int w = 0; w < 2; ++w)
    if (waves[w].state != WAVE_FREE)
      return 1;
  return 0;
}
