#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct nvcomp_fanout
  {
    const void** d_comp_ptrs;
    size_t* d_comp_sizes;
    void** d_decomp_ptrs;
    size_t* d_decomp_buf_sizes;
  };

  struct gpu_shuffle_op
  {
    void* d_buf;
    uint32_t blocksize;
    uint32_t typesize;
    uint32_t nblocks_full;
    uint32_t tail_nbytes;
  };

  struct blosc1_totals
  {
    uint32_t n_zstd;
    uint32_t n_memcpy;
    uint32_t n_parse_errors; // GPU parse, surfaced via h_parse_counters
    uint32_t n_codec_errors; // device, via decoder_status_reduce
  };

#ifdef __cplusplus
}
#endif
