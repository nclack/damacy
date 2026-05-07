#include "decoder/decoder_memcpy.h"

#include "decoder/launch_check.h"

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kThreadsPerBlock = 256;

__global__ void
gpu_memcpy_kernel(const struct gpu_memcpy_op* ops)
{
  const struct gpu_memcpy_op op = ops[blockIdx.x];
  const uint8_t* __restrict__ src = static_cast<const uint8_t*>(op.d_src);
  uint8_t* __restrict__ dst = static_cast<uint8_t*>(op.d_dst);
  const uint32_t n = op.nbytes;

  // Vectorized 16-byte copies for the aligned middle, byte tails on either
  // end. uint4 aligns to 16 bytes; the kernel falls back to byte copies
  // when the input isn't aligned.
  const uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
  const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);
  const bool aligned16 =
    ((src_addr | dst_addr) & 0xfu) == 0 && ((n & 0xfu) == 0);

  if (aligned16) {
    const uint4* src4 = reinterpret_cast<const uint4*>(src);
    uint4* dst4 = reinterpret_cast<uint4*>(dst);
    const uint32_t n4 = n >> 4;
    for (uint32_t i = threadIdx.x; i < n4; i += kThreadsPerBlock)
      dst4[i] = src4[i];
  } else {
    for (uint32_t i = threadIdx.x; i < n; i += kThreadsPerBlock)
      dst[i] = src[i];
  }
}

} // namespace

extern "C" int
decoder_memcpy_launch(CUstream stream,
                      const struct gpu_memcpy_op* d_ops,
                      uint32_t n_ops)
{
  if (n_ops == 0)
    return 0;
  gpu_memcpy_kernel<<<n_ops, kThreadsPerBlock, 0, (cudaStream_t)stream>>>(
    d_ops);
  return decoder_launch_status_check("decoder_memcpy_launch");
}
