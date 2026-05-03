#include "assemble.h"

#include <stdio.h>

namespace {

constexpr uint32_t kThreadsPerBlock = 256;

__global__ void
assemble_kernel(const struct assemble_chunk* chunks,
                uint32_t n_chunks,
                const uint8_t* arena_base,
                uint8_t* output_base,
                uint32_t bpe)
{
  uint32_t cid = blockIdx.x;
  if (cid >= n_chunks)
    return;

  const struct assemble_chunk c = chunks[cid];

  uint64_t total = 1;
  for (uint32_t d = 0; d < c.rank; ++d)
    total *= c.win[d];

  uint64_t lin = (uint64_t)blockIdx.y * blockDim.x + threadIdx.x;
  if (lin >= total)
    return;

  uint64_t src_off_elems = 0;
  uint64_t dst_off_elems = 0;
  uint64_t rem = lin;
  for (int d = (int)c.rank - 1; d >= 0; --d) {
    uint32_t w = c.win[d];
    uint32_t iter = (uint32_t)(rem % w);
    rem /= w;
    src_off_elems += (uint64_t)iter * (uint64_t)c.src_strides[d];
    dst_off_elems += (uint64_t)iter * (uint64_t)c.dst_strides[d];
  }

  const uint8_t* src = arena_base + c.src_base_byte_off + src_off_elems * bpe;
  uint8_t* dst = output_base + c.dst_base_byte_off + dst_off_elems * bpe;
  for (uint32_t b = 0; b < bpe; ++b)
    dst[b] = src[b];
}

} // namespace

extern "C" int
assemble_launch(cudaStream_t stream,
                const struct assemble_chunk* chunks_dev,
                uint32_t n_chunks,
                uint32_t max_window_elements,
                const void* arena_base,
                void* output_base,
                uint32_t bpe)
{
  if (n_chunks == 0)
    return 0;
  if (!chunks_dev || !arena_base || !output_base || bpe == 0)
    return 1;
  uint32_t blocks_y =
    (max_window_elements + kThreadsPerBlock - 1) / kThreadsPerBlock;
  if (blocks_y == 0)
    blocks_y = 1;
  dim3 grid(n_chunks, blocks_y, 1);
  dim3 block(kThreadsPerBlock, 1, 1);
  assemble_kernel<<<grid, block, 0, stream>>>(chunks_dev,
                                              n_chunks,
                                              (const uint8_t*)arena_base,
                                              (uint8_t*)output_base,
                                              bpe);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "[assemble] launch: %s\n", cudaGetErrorString(err));
    return 1;
  }
  return 0;
}
