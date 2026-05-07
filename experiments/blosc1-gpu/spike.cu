// spike.cu — derisk: does nvcomp's batched LZ4 decompress consume the
// LZ4 bytes that blosc1 produces?
//
// Reads fixture_lz4_noshuffle.blosc1 + fixture_lz4_noshuffle.raw, parses
// the blosc1 chunk to extract the first LZ4 sub-stream of the first block,
// hands it to nvcompBatchedLZ4DecompressAsync as a batch of 1, and
// byte-compares the result against raw[:blocksize/typesize].
//
// Build (from inside `nix develop`):
//   nvcc -O2 -std=c++17 -arch=sm_75 spike.cu \
//        -I"${NVCOMP_INCLUDE}" "${NVCOMP_STATIC}" \
//        -lcuda -lstdc++ -o spike
//
// where NVCOMP_INCLUDE / NVCOMP_STATIC come from the existing CMake cache
// (see build/CMakeCache.txt: NVCOMP_INCLUDE_DIR and NVCOMP_LIBRARY).
//
// Exit codes:
//   0  PASS  — decompressed bytes match expected
//   1  format mismatch / parse error
//   2  cuda or nvcomp call failed
//   3  decompressed bytes differ from expected

#include <cuda.h>
#include <nvcomp.h>
#include <nvcomp/lz4.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#define CUCHK(expr)                                                            \
  do {                                                                         \
    CUresult _r = (expr);                                                      \
    if (_r != CUDA_SUCCESS) {                                                  \
      const char* _msg = nullptr;                                              \
      cuGetErrorString(_r, &_msg);                                             \
      std::fprintf(stderr, "cuda %s -> %s\n", #expr, _msg ? _msg : "?");       \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

#define NVCHK(expr)                                                            \
  do {                                                                         \
    nvcompStatus_t _s = (expr);                                                \
    if (_s != nvcompSuccess) {                                                 \
      std::fprintf(stderr, "nvcomp %s -> %d\n", #expr, (int)_s);               \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

static std::vector<uint8_t>
slurp(const char* path)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::fprintf(stderr, "open: %s\n", path);
    std::exit(1);
  }
  std::streamsize n = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<uint8_t> v(n);
  f.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

int
main()
{
  // 1. Load fixtures.
  auto blosc = slurp("fixture_lz4_noshuffle.blosc1");
  auto raw = slurp("fixture_lz4_noshuffle.raw");

  // 2. Parse 16-byte blosc1 header.
  if (blosc.size() < 16) {
    std::fprintf(stderr, "blosc fixture too short\n");
    return 1;
  }
  uint8_t version = blosc[0];
  uint8_t versionlz = blosc[1];
  uint8_t flags = blosc[2];
  uint8_t typesize = blosc[3];
  uint32_t nbytes, blocksize, cbytes;
  std::memcpy(&nbytes, &blosc[4], 4);
  std::memcpy(&blocksize, &blosc[8], 4);
  std::memcpy(&cbytes, &blosc[12], 4);
  std::printf("header: version=%u versionlz=%u flags=0x%02x typesize=%u\n",
              version,
              versionlz,
              flags,
              typesize);
  std::printf(
    "        nbytes=%u blocksize=%u cbytes=%u\n", nbytes, blocksize, cbytes);

  // Sanity: not memcpyed (flag bit 1), compformat = LZ4 (bits 5..7 = 1).
  if (flags & 0x02) {
    std::fprintf(stderr, "fixture is MEMCPYED — useless for LZ4 spike\n");
    return 1;
  }
  uint8_t compformat = (flags >> 5) & 0x07;
  if (compformat != 1) {
    std::fprintf(
      stderr, "fixture compformat=%u, expected 1 (LZ4)\n", compformat);
    return 1;
  }

  // 3. Parse offset table — one int32 per block.
  uint32_t nblocks = (nbytes + blocksize - 1) / blocksize;
  if (16 + 4 * nblocks > blosc.size()) {
    std::fprintf(stderr, "offset table runs past end of fixture\n");
    return 1;
  }
  std::vector<int32_t> bstarts(nblocks);
  std::memcpy(bstarts.data(), &blosc[16], 4 * nblocks);

  // 4. Locate block 0; for typesize > 1 with split (the blosc1 default for
  //    LZ4 with non-trivial blocksize), the block is split into `typesize`
  //    sub-streams, each prefixed by an int32 cbytes. Pull sub-stream 0.
  if (nblocks < 1) {
    std::fprintf(stderr, "no blocks\n");
    return 1;
  }
  size_t block0_off = (size_t)bstarts[0];
  size_t block0_end = (nblocks > 1) ? (size_t)bstarts[1] : cbytes;
  std::printf("block 0: payload @ %zu..%zu (size=%zu)\n",
              block0_off,
              block0_end,
              block0_end - block0_off);

  // Sub-stream 0: int32 prefix at block0_off, then sub_cbytes bytes.
  if (block0_off + 4 > blosc.size()) {
    std::fprintf(stderr, "block 0 truncated\n");
    return 1;
  }
  int32_t sub0_cbytes;
  std::memcpy(&sub0_cbytes, &blosc[block0_off], 4);
  size_t sub0_off = block0_off + 4;
  size_t sub0_end = sub0_off + sub0_cbytes;
  if (sub0_end > blosc.size() || sub0_cbytes <= 0) {
    std::fprintf(stderr, "sub-stream 0 size %d implausible\n", sub0_cbytes);
    return 1;
  }
  uint32_t sub0_decompressed_bytes = blocksize / typesize;
  std::printf("sub-stream 0: lz4 bytes @ %zu..%zu (csize=%d) -> %u dsize\n",
              sub0_off,
              sub0_end,
              sub0_cbytes,
              sub0_decompressed_bytes);

  // 5. Set up CUDA: get a context.
  CUCHK(cuInit(0));
  CUdevice dev;
  CUCHK(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  CUCHK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CUCHK(cuCtxSetCurrent(ctx));

  CUstream stream;
  CUCHK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

  // 6. Allocate device buffers and upload sub-stream 0.
  CUdeviceptr d_compressed = 0;
  CUdeviceptr d_decompressed = 0;
  CUCHK(cuMemAlloc(&d_compressed, (size_t)sub0_cbytes));
  CUCHK(cuMemAlloc(&d_decompressed, (size_t)sub0_decompressed_bytes));
  CUCHK(cuMemcpyHtoDAsync(
    d_compressed, &blosc[sub0_off], (size_t)sub0_cbytes, stream));

  // 7. nvcomp's batched API needs *device* arrays of pointers and sizes.
  //    Build a one-element batch on host, copy to device.
  void* h_comp_ptr = (void*)(uintptr_t)d_compressed;
  size_t h_comp_size = (size_t)sub0_cbytes;
  size_t h_decomp_buf_size = (size_t)sub0_decompressed_bytes;
  void* h_decomp_ptr = (void*)(uintptr_t)d_decompressed;

  CUdeviceptr d_comp_ptrs = 0, d_comp_sizes = 0;
  CUdeviceptr d_decomp_buf_sizes = 0, d_decomp_actual_sizes = 0;
  CUdeviceptr d_decomp_ptrs = 0;
  CUdeviceptr d_statuses = 0;
  CUCHK(cuMemAlloc(&d_comp_ptrs, sizeof(void*)));
  CUCHK(cuMemAlloc(&d_comp_sizes, sizeof(size_t)));
  CUCHK(cuMemAlloc(&d_decomp_buf_sizes, sizeof(size_t)));
  CUCHK(cuMemAlloc(&d_decomp_actual_sizes, sizeof(size_t)));
  CUCHK(cuMemAlloc(&d_decomp_ptrs, sizeof(void*)));
  CUCHK(cuMemAlloc(&d_statuses, sizeof(nvcompStatus_t)));

  CUCHK(cuMemcpyHtoDAsync(d_comp_ptrs, &h_comp_ptr, sizeof(void*), stream));
  CUCHK(cuMemcpyHtoDAsync(d_comp_sizes, &h_comp_size, sizeof(size_t), stream));
  CUCHK(cuMemcpyHtoDAsync(
    d_decomp_buf_sizes, &h_decomp_buf_size, sizeof(size_t), stream));
  CUCHK(cuMemcpyHtoDAsync(d_decomp_ptrs, &h_decomp_ptr, sizeof(void*), stream));

  // 8. Query temp size, allocate, then call decompress.
  nvcompBatchedLZ4DecompressOpts_t opts = nvcompBatchedLZ4DecompressDefaultOpts;
  size_t temp_bytes = 0;
  NVCHK(nvcompBatchedLZ4DecompressGetTempSizeAsync(
    1, sub0_decompressed_bytes, opts, &temp_bytes, sub0_decompressed_bytes));
  std::printf("nvcomp temp: %zu bytes\n", temp_bytes);
  CUdeviceptr d_temp = 0;
  if (temp_bytes > 0)
    CUCHK(cuMemAlloc(&d_temp, temp_bytes));

  NVCHK(nvcompBatchedLZ4DecompressAsync(
    reinterpret_cast<const void* const*>((void*)(uintptr_t)d_comp_ptrs),
    reinterpret_cast<const size_t*>((void*)(uintptr_t)d_comp_sizes),
    reinterpret_cast<const size_t*>((void*)(uintptr_t)d_decomp_buf_sizes),
    reinterpret_cast<size_t*>((void*)(uintptr_t)d_decomp_actual_sizes),
    1,
    reinterpret_cast<void*>((void*)(uintptr_t)d_temp),
    temp_bytes,
    reinterpret_cast<void* const*>((void*)(uintptr_t)d_decomp_ptrs),
    opts,
    reinterpret_cast<nvcompStatus_t*>((void*)(uintptr_t)d_statuses),
    stream));

  CUCHK(cuStreamSynchronize(stream));

  // 9. Pull the status and the decompressed bytes back, compare.
  nvcompStatus_t status = nvcompErrorInternal;
  CUCHK(cuMemcpyDtoH(&status, d_statuses, sizeof(status)));
  size_t actual_size = 0;
  CUCHK(cuMemcpyDtoH(&actual_size, d_decomp_actual_sizes, sizeof(actual_size)));
  std::printf("decompress status=%d, actual_size=%zu (expected %u)\n",
              (int)status,
              actual_size,
              sub0_decompressed_bytes);
  if (status != nvcompSuccess) {
    std::fprintf(stderr, "FAIL: nvcomp returned non-success status\n");
    return 3;
  }
  if (actual_size != sub0_decompressed_bytes) {
    std::fprintf(stderr, "FAIL: actual_size mismatch\n");
    return 3;
  }

  std::vector<uint8_t> got(sub0_decompressed_bytes);
  CUCHK(cuMemcpyDtoH(got.data(), d_decompressed, sub0_decompressed_bytes));

  // Expected bytes: with shuffle=NOSHUFFLE and split=ON, sub-stream 0 of
  // block 0 contains the *first* blocksize/typesize bytes of the original.
  if (raw.size() < sub0_decompressed_bytes) {
    std::fprintf(stderr, "raw fixture shorter than expected slice\n");
    return 1;
  }
  int mismatch = std::memcmp(got.data(), raw.data(), sub0_decompressed_bytes);
  if (mismatch != 0) {
    // Find first differing byte.
    size_t i = 0;
    while (i < sub0_decompressed_bytes && got[i] == raw[i])
      ++i;
    std::fprintf(stderr,
                 "FAIL: bytes differ at offset %zu — got 0x%02x, want 0x%02x\n",
                 i,
                 got[i],
                 raw[i]);
    return 3;
  }

  std::printf("PASS: %u bytes match expected slice\n", sub0_decompressed_bytes);
  return 0;
}
