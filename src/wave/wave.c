#include "wave.h"

#include "damacy_limits.h"
#include "decoder/blosc1.h"
#include "decoder/decoder_memcpy.h"
#include "fanout.h"
#include "util/cuda_check.h"
#include "util/prelude.h"
#include "wave/input_slot.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

int
wave_init(struct damacy_wave* wave,
          uint32_t max_chunks_per_wave,
          uint32_t max_substreams_per_wave,
          uint64_t input_staging_device_bytes,
          uint64_t dev_decompressed_bytes)
{
  memset(wave, 0, sizeof(*wave));
  wave->state = WAVE_FREE;
  wave->bound_slot = -1;
  wave->dev_decompressed_cap = dev_decompressed_bytes;

  CUdeviceptr dptr = 0;
  if (input_staging_device_bytes > 0) {
    CU(Error, cuMemAlloc(&dptr, input_staging_device_bytes));
    wave->dev_compressed_owned = (void*)(uintptr_t)dptr;
    wave->dev_compressed = wave->dev_compressed_owned;
  }
  CU(Error, cuMemAlloc(&dptr, dev_decompressed_bytes));
  wave->dev_decompressed = (void*)(uintptr_t)dptr;

  uint32_t cap = max_chunks_per_wave;
  CU(Error,
     cuMemAllocHost((void**)&wave->h_blosc1_totals,
                    sizeof(struct blosc1_totals)));
  CU(Error,
     cuMemAllocHost((void**)&wave->h_assemble_chunks,
                    (size_t)cap * sizeof(struct assemble_chunk)));

  CU(Error, cuMemAlloc(&dptr, (size_t)cap * sizeof(struct assemble_chunk)));
  wave->d_assemble_chunks = (struct assemble_chunk*)(uintptr_t)dptr;

  CU(Error, cuMemAlloc(&dptr, sizeof(struct blosc1_totals)));
  wave->d_blosc1_totals = (struct blosc1_totals*)(uintptr_t)dptr;

  CU(Error,
     cuMemAllocHost((void**)&wave->h_parse_chunks,
                    (size_t)cap * sizeof(struct gpu_parse_chunk)));
  CU(Error, cuMemAlloc(&dptr, (size_t)cap * sizeof(struct gpu_parse_chunk)));
  wave->d_parse_chunks = (struct gpu_parse_chunk*)(uintptr_t)dptr;
  CU(Error,
     cuMemAllocHost((void**)&wave->h_blosc_chunk_indices,
                    (size_t)cap * sizeof(uint32_t)));
  CU(Error, cuMemAlloc(&dptr, (size_t)cap * sizeof(uint32_t)));
  wave->d_blosc_chunk_indices = (uint32_t*)(uintptr_t)dptr;
  {
    const size_t substreams_max = max_substreams_per_wave;
    CU(Error,
       cuMemAllocHost((void**)&wave->h_block_chunk_map,
                      substreams_max * sizeof(uint32_t)));
    CU(Error, cuMemAlloc(&dptr, substreams_max * sizeof(uint32_t)));
  }
  wave->d_block_chunk_map = (uint32_t*)(uintptr_t)dptr;
  // Bitset: one bit per wave-local chunk index, rounded to uint32_t words.
  CU(Error, cuMemAlloc(&dptr, (size_t)((cap + 31u) / 32u) * sizeof(uint32_t)));
  wave->d_is_memcpyed = (uint32_t*)(uintptr_t)dptr;
  CU(Error, cuMemAlloc(&dptr, sizeof(uint32_t)));
  wave->d_n_zstd = (uint32_t*)(uintptr_t)dptr;
  CU(Error, cuMemAlloc(&dptr, sizeof(uint32_t)));
  wave->d_n_memcpy = (uint32_t*)(uintptr_t)dptr;
  CU(Error, cuMemAlloc(&dptr, sizeof(uint32_t)));
  wave->d_parse_err = (uint32_t*)(uintptr_t)dptr;
  CU(Error,
     cuMemAllocHost((void**)&wave->h_parse_counters, 3 * sizeof(uint32_t)));

  const size_t substreams = DAMACY_BLOSC_ZSTD_INITIAL_BATCH_CAP;
  if (fanout_alloc_pinned(&wave->h_zstd_fan, &wave->zstd_fan, substreams))
    goto Error;
  wave->fanout_cap = (uint32_t)substreams;

  CU(Error,
     cuMemAllocHost((void**)&wave->h_memcpy_ops,
                    (size_t)cap * sizeof(struct gpu_memcpy_op)));
  CU(Error, cuMemAlloc(&dptr, (size_t)cap * sizeof(struct gpu_memcpy_op)));
  wave->d_memcpy_ops = (struct gpu_memcpy_op*)(uintptr_t)dptr;

  CU(Error, cuEventCreate(&wave->ev.input_start, CU_EVENT_DEFAULT));
  CU(Error, cuEventCreate(&wave->ev.input_transfer_done, CU_EVENT_DEFAULT));
  CU(Error, cuEventCreate(&wave->ev.input_parse_done, CU_EVENT_DEFAULT));
  CU(Error, cuEventCreate(&wave->ev.decomp_start, CU_EVENT_DEFAULT));
  CU(Error, cuEventCreate(&wave->ev.decode_done, CU_EVENT_DEFAULT));
  CU(Error, cuEventCreate(&wave->ev.decomp_end, CU_EVENT_DEFAULT));
  CU(Error, cuEventCreate(&wave->ev.asm_start, CU_EVENT_DEFAULT));
  CU(Error, cuEventCreate(&wave->ev.asm_end, CU_EVENT_DEFAULT));

  return 0;
Error:
  wave_destroy(wave, 0);
  return 1;
}

void
wave_destroy(struct damacy_wave* wave, int cuda_skip)
{
  if (!wave)
    return;
  if (!cuda_skip) {
    void* const host_ptrs[] = {
      wave->h_blosc1_totals,   wave->h_memcpy_ops,
      wave->h_assemble_chunks, wave->h_parse_chunks,
      wave->h_parse_counters,  wave->h_blosc_chunk_indices,
      wave->h_block_chunk_map,
    };
    for (size_t i = 0; i < countof(host_ptrs); ++i)
      if (host_ptrs[i])
        cuMemFreeHost(host_ptrs[i]);
    fanout_free_pinned(&wave->h_zstd_fan, &wave->zstd_fan);
    void* const dev_ptrs[] = {
      wave->dev_compressed_owned,
      wave->dev_decompressed,
      wave->d_assemble_chunks,
      wave->d_blosc1_totals,
      wave->d_memcpy_ops,
      wave->d_parse_chunks,
      wave->d_blosc_chunk_indices,
      wave->d_block_chunk_map,
      wave->d_is_memcpyed,
      wave->d_n_zstd,
      wave->d_n_memcpy,
      wave->d_parse_err,
    };
    for (size_t i = 0; i < countof(dev_ptrs); ++i)
      if (dev_ptrs[i])
        cuMemFree(CUDPTR(dev_ptrs[i]));
    CUevent* const events[] = {
      &wave->ev.input_start,      &wave->ev.input_transfer_done,
      &wave->ev.input_parse_done, &wave->ev.decomp_start,
      &wave->ev.decode_done,      &wave->ev.decomp_end,
      &wave->ev.asm_start,        &wave->ev.asm_end
    };
    for (size_t i = 0; i < countof(events); ++i)
      if (*events[i])
        cuEventDestroy_v2(*events[i]);
  }
  memset(wave, 0, sizeof(*wave));
}

void
wave_bind_input_slot(struct damacy_wave* wave,
                     int slot_idx,
                     const struct input_slot* slot,
                     void* dev_compressed)
{
  wave->bound_slot = (int8_t)slot_idx;
  wave->host_input = slot->buf;
  wave->dev_compressed = dev_compressed;
  wave->render_job_idx = slot->render_job_idx;
  wave->batch_pool_slot = slot->batch_pool_slot;
  wave->batch_chunk_offset = slot->batch_chunk_offset;
  wave->n_chunks = slot->n_chunks;
  wave->input_used_bytes = slot->used_bytes;
  wave->io_bytes = slot->io_bytes;
  wave->io_ms = slot->io_ms;
  wave->decomp_in_bytes = 0;
  wave->decomp_out_bytes = 0;
  wave->assemble_out_bytes = 0;
}

void
wave_unbind_input_slot(struct damacy_wave* wave)
{
  wave->bound_slot = -1;
  wave->host_input = NULL;
  wave->dev_compressed = wave->dev_compressed_owned;
}
