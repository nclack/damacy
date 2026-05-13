#include "decoder/decoder_memcpy.h"
#include "decoder/decoder_zstd.h"
#include "expect.h"

#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int
test_create_destroy(void)
{
  struct decoder_zstd* z = decoder_zstd_create(8, 64 * 1024, 8 * 64 * 1024);
  EXPECT(z);
  decoder_zstd_destroy(z);
  decoder_zstd_destroy(NULL);

  return 0;
}

// Run a small batch of memcpy ops on the GPU and verify byte-for-byte.
// 3 ops: aligned-16 sized, unaligned size, and unaligned-source pointer.
static int
test_memcpy_kernel(void)
{
  const size_t kArenaBytes = 4096;
  const size_t kOpsCount = 3;
  const uint32_t op_sizes[3] = { 256, 257, 200 };
  const uint32_t op_src_offs[3] = { 0, 256, 514 };
  const uint32_t op_dst_offs[3] = { 0, 1024, 2048 };

  uint8_t* h_src = (uint8_t*)malloc(kArenaBytes);
  uint8_t* h_dst = (uint8_t*)calloc(kArenaBytes, 1);
  uint8_t* h_dst_check = (uint8_t*)calloc(kArenaBytes, 1);
  EXPECT(h_src && h_dst && h_dst_check);
  for (size_t i = 0; i < kArenaBytes; ++i)
    h_src[i] = (uint8_t)((i * 31u + 7u) & 0xff);

  CUdeviceptr d_src = 0, d_dst = 0, d_ops = 0;
  EXPECT(cuMemAlloc(&d_src, kArenaBytes) == CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_dst, kArenaBytes) == CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_ops, kOpsCount * sizeof(struct gpu_memcpy_op)) ==
         CUDA_SUCCESS);
  EXPECT(cuMemcpyHtoD(d_src, h_src, kArenaBytes) == CUDA_SUCCESS);
  EXPECT(cuMemsetD8(d_dst, 0, kArenaBytes) == CUDA_SUCCESS);

  struct gpu_memcpy_op h_ops[3];
  for (size_t i = 0; i < kOpsCount; ++i) {
    h_ops[i].d_src = (const void*)(uintptr_t)(d_src + op_src_offs[i]);
    h_ops[i].d_dst = (void*)(uintptr_t)(d_dst + op_dst_offs[i]);
    h_ops[i].nbytes = op_sizes[i];
  }
  EXPECT(cuMemcpyHtoD(d_ops, h_ops, sizeof h_ops) == CUDA_SUCCESS);

  CUstream stream = 0;
  EXPECT(cuStreamCreate(&stream, CU_STREAM_DEFAULT) == CUDA_SUCCESS);
  EXPECT(decoder_memcpy_launch(stream,
                               (const struct gpu_memcpy_op*)(uintptr_t)d_ops,
                               kOpsCount) == 0);
  EXPECT(cuStreamSynchronize(stream) == CUDA_SUCCESS);

  EXPECT(cuMemcpyDtoH(h_dst_check, d_dst, kArenaBytes) == CUDA_SUCCESS);
  for (size_t i = 0; i < kOpsCount; ++i) {
    const uint8_t* expect = h_src + op_src_offs[i];
    const uint8_t* got = h_dst_check + op_dst_offs[i];
    for (uint32_t b = 0; b < op_sizes[i]; ++b)
      EXPECT(expect[b] == got[b]);
  }

  cuStreamDestroy(stream);
  cuMemFree(d_src);
  cuMemFree(d_dst);
  cuMemFree(d_ops);
  free(h_src);
  free(h_dst);
  free(h_dst_check);
  return 0;
}

int
main(void)
{
  EXPECT(cuInit(0) == CUDA_SUCCESS);
  CUdevice dev = 0;
  EXPECT(cuDeviceGet(&dev, 0) == CUDA_SUCCESS);
  CUcontext ctx = NULL;
  EXPECT(cuDevicePrimaryCtxRetain(&ctx, dev) == CUDA_SUCCESS);
  EXPECT(cuCtxSetCurrent(ctx) == CUDA_SUCCESS);

  RUN(test_create_destroy);
  RUN(test_memcpy_kernel);

  cuDevicePrimaryCtxRelease(dev);
  log_info("all tests passed");
  return 0;
}
