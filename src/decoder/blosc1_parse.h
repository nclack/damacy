#pragma once

#include "decoder/blosc1.h"
#include "decoder/decoder_memcpy.h"

#include <stdint.h>

typedef struct CUstream_st* CUstream;

#ifdef __cplusplus
extern "C"
{
#endif

  struct assemble_chunk;
  struct sample_plan;

  struct gpu_parse_chunk
  {
    uint32_t compressed_offset;
    uint32_t compressed_nbytes;
    uint32_t decompressed_offset;
    uint32_t decompressed_nbytes;
    uint16_t sample_idx_in_batch;
    uint8_t codec_id;
    uint8_t is_fill;
  };

  struct blosc1_parse_args
  {
    const uint8_t* d_compressed;
    uint8_t* d_decompressed;
    const struct gpu_parse_chunk* d_chunks;
    const struct sample_plan* d_sample_plans;
    uint32_t n_sample_plans;
    const uint32_t* d_blosc_chunk_indices;
    uint32_t n_blosc_chunks;
    const uint32_t* d_block_chunk_map;
    uint32_t n_blosc_blocks;
    uint32_t* d_skip_blocks;
    struct decode_fanout fanout;
    uint32_t fanout_cap;
    struct assemble_chunk* d_assemble_chunks;
    uint32_t* d_n_zstd;
    uint32_t* d_n_lz4;
    uint32_t* d_n_memcpy;
    uint32_t* d_parse_err;
  };

  int blosc1_parse_launch(CUstream stream,
                          const struct blosc1_parse_args* args);

#ifdef __cplusplus
}
#endif
