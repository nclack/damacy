// spike.cu — GPU blosc1 decompress derisk: matrix runner.
//
// Usage: ./spike <fixture-base-name>
//
// Reads <name>.blosc1 + <name>.raw, parses the blosc1 chunk, dispatches
// the appropriate nvcomp batched LZ4 or Zstd call across all sub-streams
// of all blocks, optionally CPU-unshuffles, and byte-compares to .raw.
//
// Build (from inside `nix develop`):
//   nvcc -O2 -std=c++17 -arch=sm_75 spike.cu \
//        -I"$NVCOMP_INC" "$NVCOMP_LIB" -lcuda -o spike
//
// Exit codes:
//   0  PASS
//   1  parse / setup error
//   2  cuda or nvcomp call failed
//   3  byte mismatch

#include <cuda.h>
#include <nvcomp.h>
#include <nvcomp/lz4.h>
#include <nvcomp/zstd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
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
slurp(const std::string& path)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::fprintf(stderr, "open: %s\n", path.c_str());
    std::exit(1);
  }
  std::streamsize n = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<uint8_t> v(n);
  f.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

struct BloscHdr
{
  uint8_t version, versionlz, flags, typesize;
  uint32_t nbytes, blocksize, cbytes;
  uint32_t nblocks;
  uint8_t compformat; // (flags >> 5) & 7  → 1=LZ4 4=ZSTD
  bool shuffle;       // flags & 1
  bool memcpyed;      // flags & 2
};

// One LZ4/Zstd unit handed to nvcomp.
struct SubStream
{
  size_t src_off;  // offset into compressed buffer (just the LZ4/zstd bytes)
  size_t src_size; // size of LZ4/zstd bytes
  size_t dst_off;  // offset into decompressed buffer
  size_t dst_size; // expected decompressed size
};

static void
parse_header(const std::vector<uint8_t>& b, BloscHdr& h)
{
  if (b.size() < 16) {
    std::fprintf(stderr, "blosc fixture too short\n");
    std::exit(1);
  }
  h.version = b[0];
  h.versionlz = b[1];
  h.flags = b[2];
  h.typesize = b[3];
  std::memcpy(&h.nbytes, &b[4], 4);
  std::memcpy(&h.blocksize, &b[8], 4);
  std::memcpy(&h.cbytes, &b[12], 4);
  h.nblocks = (h.nbytes + h.blocksize - 1) / h.blocksize;
  h.compformat = (h.flags >> 5) & 0x07;
  h.shuffle = (h.flags & 0x01) != 0;
  h.memcpyed = (h.flags & 0x02) != 0;
}

// Build the flat sub-stream table by walking the offset table + per-block
// payloads. Assumes uniform blocks (nbytes % blocksize == 0).
//
// Real blosc1 layout (verified empirically against numcodecs.Blosc):
//   - bstarts[bi] gives block bi's payload start in the blob; values are
//     NOT in block-index order (writer threads finish out of order, so
//     blocks are scattered across the payload region in completion order).
//     To find a block's end-of-payload we sort bstarts and use the next
//     larger offset (or cbytes for the largest).
//   - Each block contains 1+ sub-streams; each sub-stream has an int32
//     cbytes prefix followed by that many codec bytes. The sub-stream
//     count is implicit and codec-dependent:
//       lz4 / lz4hc with typesize > 1 (and "non-trivial" blocksize) → typesize
//       sub-streams lz4 / lz4hc with typesize = 1 → 1 sub-stream zstd → 1
//       sub-stream (no split)
//     We discover this by walking until the accumulated size matches the
//     block's compressed payload, and assert the count is 1 or typesize.
static std::vector<SubStream>
build_table(const std::vector<uint8_t>& blob,
            const BloscHdr& h,
            uint32_t& nstreams_per_block)
{
  std::vector<SubStream> out;

  std::vector<int32_t> bstarts(h.nblocks);
  std::memcpy(bstarts.data(), &blob[16], 4 * h.nblocks);

  // Build a sorted list of (offset, block_index) so we can find each
  // block's end-of-payload regardless of writer thread order.
  std::vector<std::pair<size_t, uint32_t>> sorted;
  sorted.reserve(h.nblocks);
  for (uint32_t bi = 0; bi < h.nblocks; ++bi)
    sorted.emplace_back((size_t)bstarts[bi], bi);
  std::sort(sorted.begin(), sorted.end());

  // Map: block_index -> end_offset.
  std::vector<size_t> block_end(h.nblocks);
  for (size_t k = 0; k < sorted.size(); ++k) {
    size_t end =
      (k + 1 < sorted.size()) ? sorted[k + 1].first : (size_t)h.cbytes;
    block_end[sorted[k].second] = end;
  }

  nstreams_per_block = 0;
  for (uint32_t bi = 0; bi < h.nblocks; ++bi) {
    size_t off = (size_t)bstarts[bi];
    size_t end = block_end[bi];
    size_t block_dst = (size_t)bi * h.blocksize;

    // Walk sub-streams; count them.
    std::vector<std::pair<size_t, size_t>> subs; // (src_off, src_size)
    size_t cur = off;
    while (cur < end) {
      if (cur + 4 > blob.size()) {
        std::fprintf(stderr, "block %u: truncated prefix at %zu\n", bi, cur);
        std::exit(1);
      }
      int32_t cb;
      std::memcpy(&cb, &blob[cur], 4);
      cur += 4;
      if (cb < 0 || cur + (size_t)cb > end) {
        std::fprintf(stderr,
                     "block %u: implausible sub-stream cbytes=%d at %zu\n",
                     bi,
                     cb,
                     cur);
        std::exit(1);
      }
      subs.emplace_back(cur, (size_t)cb);
      cur += (size_t)cb;
    }
    if (cur != end) {
      std::fprintf(
        stderr, "block %u: walk ended at %zu, expected %zu\n", bi, cur, end);
      std::exit(1);
    }

    // The first block sets nstreams_per_block; subsequent blocks must agree.
    if (bi == 0) {
      nstreams_per_block = (uint32_t)subs.size();
      if (nstreams_per_block != 1 && nstreams_per_block != h.typesize) {
        std::fprintf(stderr,
                     "block 0: nstreams=%u not in {1, typesize=%u}\n",
                     nstreams_per_block,
                     h.typesize);
        std::exit(1);
      }
    } else if (subs.size() != nstreams_per_block) {
      std::fprintf(stderr,
                   "block %u: nstreams=%zu disagrees with first %u\n",
                   bi,
                   subs.size(),
                   nstreams_per_block);
      std::exit(1);
    }

    // Sub-stream dst layout:
    //   nstreams==1            → one stream covering the whole block
    //   nstreams==typesize     → split: equal slices
    size_t per_stream_dsize =
      (nstreams_per_block == 1) ? h.blocksize : (h.blocksize / h.typesize);
    for (size_t i = 0; i < subs.size(); ++i) {
      SubStream s{};
      s.src_off = subs[i].first;
      s.src_size = subs[i].second;
      s.dst_off = block_dst + i * per_stream_dsize;
      s.dst_size = per_stream_dsize;
      out.push_back(s);
    }
  }
  return out;
}

// Per-block byte unshuffle (CPU). Reverses blosc1's byte-shuffle filter:
// shuffled layout has all byte-0 first, then all byte-1, etc. Reconstruct
// the natural [N][typesize] interleaving in place.
static void
unshuffle_blocks(uint8_t* buf, const BloscHdr& h)
{
  std::vector<uint8_t> tmp(h.blocksize);
  uint32_t N = h.blocksize / h.typesize;
  for (uint32_t bi = 0; bi < h.nblocks; ++bi) {
    uint8_t* block = buf + (size_t)bi * h.blocksize;
    std::memcpy(tmp.data(), block, h.blocksize);
    for (uint32_t i = 0; i < N; ++i) {
      for (uint32_t b = 0; b < h.typesize; ++b) {
        block[i * h.typesize + b] = tmp[(size_t)b * N + i];
      }
    }
  }
}

int
main(int argc, char** argv)
{
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <fixture-base-name>\n", argv[0]);
    return 1;
  }
  std::string base = argv[1];
  auto blob = slurp(base + ".blosc1");
  auto raw = slurp(base + ".raw");

  BloscHdr h{};
  parse_header(blob, h);
  std::printf(
    "[%s] flags=0x%02x compformat=%u shuffle=%d memcpyed=%d typesize=%u\n",
    base.c_str(),
    h.flags,
    h.compformat,
    (int)h.shuffle,
    (int)h.memcpyed,
    h.typesize);
  std::printf("       nbytes=%u blocksize=%u cbytes=%u nblocks=%u\n",
              h.nbytes,
              h.blocksize,
              h.cbytes,
              h.nblocks);

  if (h.memcpyed) {
    std::fprintf(stderr, "FAIL: fixture is MEMCPYED — outside scope\n");
    return 1;
  }
  if (h.compformat != 1 && h.compformat != 4) {
    std::fprintf(stderr, "FAIL: compformat=%u not LZ4 or zstd\n", h.compformat);
    return 1;
  }
  if (h.nbytes % h.blocksize != 0) {
    std::fprintf(stderr, "FAIL: spike requires nbytes %% blocksize == 0\n");
    return 1;
  }

  uint32_t nstreams_per_block = 0;
  auto table = build_table(blob, h, nstreams_per_block);
  std::printf("       nstreams/block=%u total sub-streams=%zu\n",
              nstreams_per_block,
              table.size());

  // CUDA setup.
  CUCHK(cuInit(0));
  CUdevice dev;
  CUCHK(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  CUCHK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CUCHK(cuCtxSetCurrent(ctx));
  CUstream stream;
  CUCHK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

  // Upload the entire blosc blob (we'll address sub-streams by offset).
  CUdeviceptr d_blob = 0;
  CUCHK(cuMemAlloc(&d_blob, blob.size()));
  CUCHK(cuMemcpyHtoDAsync(d_blob, blob.data(), blob.size(), stream));

  // Decompressed output: nbytes contiguous bytes.
  CUdeviceptr d_out = 0;
  CUCHK(cuMemAlloc(&d_out, h.nbytes));

  // Build per-sub-stream device pointer / size arrays.
  size_t n = table.size();
  std::vector<const void*> h_comp_ptrs(n);
  std::vector<size_t> h_comp_sizes(n);
  std::vector<void*> h_decomp_ptrs(n);
  std::vector<size_t> h_decomp_buf_sizes(n);
  size_t max_decompressed = 0;
  for (size_t i = 0; i < n; ++i) {
    h_comp_ptrs[i] =
      reinterpret_cast<const void*>(d_blob + (CUdeviceptr)table[i].src_off);
    h_comp_sizes[i] = table[i].src_size;
    h_decomp_ptrs[i] =
      reinterpret_cast<void*>(d_out + (CUdeviceptr)table[i].dst_off);
    h_decomp_buf_sizes[i] = table[i].dst_size;
    if (table[i].dst_size > max_decompressed)
      max_decompressed = table[i].dst_size;
  }

  // Allocate device-side fanout arrays + statuses + actual-size output.
  CUdeviceptr d_comp_ptrs = 0, d_comp_sizes = 0;
  CUdeviceptr d_decomp_buf_sizes = 0, d_decomp_actual_sizes = 0;
  CUdeviceptr d_decomp_ptrs = 0, d_statuses = 0;
  CUCHK(cuMemAlloc(&d_comp_ptrs, n * sizeof(void*)));
  CUCHK(cuMemAlloc(&d_comp_sizes, n * sizeof(size_t)));
  CUCHK(cuMemAlloc(&d_decomp_buf_sizes, n * sizeof(size_t)));
  CUCHK(cuMemAlloc(&d_decomp_actual_sizes, n * sizeof(size_t)));
  CUCHK(cuMemAlloc(&d_decomp_ptrs, n * sizeof(void*)));
  CUCHK(cuMemAlloc(&d_statuses, n * sizeof(nvcompStatus_t)));
  CUCHK(cuMemcpyHtoDAsync(
    d_comp_ptrs, h_comp_ptrs.data(), n * sizeof(void*), stream));
  CUCHK(cuMemcpyHtoDAsync(
    d_comp_sizes, h_comp_sizes.data(), n * sizeof(size_t), stream));
  CUCHK(cuMemcpyHtoDAsync(
    d_decomp_buf_sizes, h_decomp_buf_sizes.data(), n * sizeof(size_t), stream));
  CUCHK(cuMemcpyHtoDAsync(
    d_decomp_ptrs, h_decomp_ptrs.data(), n * sizeof(void*), stream));

  // Dispatch.
  if (h.compformat == 1) {
    auto opts = nvcompBatchedLZ4DecompressDefaultOpts;
    size_t temp_bytes = 0;
    NVCHK(nvcompBatchedLZ4DecompressGetTempSizeAsync(
      n, max_decompressed, opts, &temp_bytes, h.nbytes));
    CUdeviceptr d_temp = 0;
    if (temp_bytes > 0)
      CUCHK(cuMemAlloc(&d_temp, temp_bytes));
    NVCHK(nvcompBatchedLZ4DecompressAsync(
      reinterpret_cast<const void* const*>(d_comp_ptrs),
      reinterpret_cast<const size_t*>(d_comp_sizes),
      reinterpret_cast<const size_t*>(d_decomp_buf_sizes),
      reinterpret_cast<size_t*>(d_decomp_actual_sizes),
      n,
      reinterpret_cast<void*>(d_temp),
      temp_bytes,
      reinterpret_cast<void* const*>(d_decomp_ptrs),
      opts,
      reinterpret_cast<nvcompStatus_t*>(d_statuses),
      stream));
  } else { // zstd
    auto opts = nvcompBatchedZstdDecompressDefaultOpts;
    size_t temp_bytes = 0;
    NVCHK(nvcompBatchedZstdDecompressGetTempSizeAsync(
      n, max_decompressed, opts, &temp_bytes, h.nbytes));
    CUdeviceptr d_temp = 0;
    if (temp_bytes > 0)
      CUCHK(cuMemAlloc(&d_temp, temp_bytes));
    NVCHK(nvcompBatchedZstdDecompressAsync(
      reinterpret_cast<const void* const*>(d_comp_ptrs),
      reinterpret_cast<const size_t*>(d_comp_sizes),
      reinterpret_cast<const size_t*>(d_decomp_buf_sizes),
      reinterpret_cast<size_t*>(d_decomp_actual_sizes),
      n,
      reinterpret_cast<void*>(d_temp),
      temp_bytes,
      reinterpret_cast<void* const*>(d_decomp_ptrs),
      opts,
      reinterpret_cast<nvcompStatus_t*>(d_statuses),
      stream));
  }
  CUCHK(cuStreamSynchronize(stream));

  // Pull statuses + actual sizes.
  std::vector<nvcompStatus_t> statuses(n);
  std::vector<size_t> actuals(n);
  CUCHK(cuMemcpyDtoH(statuses.data(), d_statuses, n * sizeof(nvcompStatus_t)));
  CUCHK(
    cuMemcpyDtoH(actuals.data(), d_decomp_actual_sizes, n * sizeof(size_t)));
  for (size_t i = 0; i < n; ++i) {
    if (statuses[i] != nvcompSuccess) {
      std::fprintf(
        stderr, "FAIL: sub-stream %zu status=%d\n", i, (int)statuses[i]);
      return 3;
    }
    if (actuals[i] != table[i].dst_size) {
      std::fprintf(stderr,
                   "FAIL: sub-stream %zu actual=%zu expected=%zu\n",
                   i,
                   actuals[i],
                   table[i].dst_size);
      return 3;
    }
  }

  // Pull decompressed bytes.
  std::vector<uint8_t> got(h.nbytes);
  CUCHK(cuMemcpyDtoH(got.data(), d_out, h.nbytes));

  // CPU un-shuffle if the chunk had shuffle on (per-block transpose).
  if (h.shuffle)
    unshuffle_blocks(got.data(), h);

  if (raw.size() < h.nbytes) {
    std::fprintf(
      stderr, "FAIL: raw fixture %zu < nbytes %u\n", raw.size(), h.nbytes);
    return 1;
  }
  if (std::memcmp(got.data(), raw.data(), h.nbytes) != 0) {
    size_t i = 0;
    while (i < h.nbytes && got[i] == raw[i])
      ++i;
    std::fprintf(stderr,
                 "FAIL: bytes differ at offset %zu — got 0x%02x want 0x%02x\n",
                 i,
                 got[i],
                 raw[i]);
    return 3;
  }

  std::printf("       PASS  (%u bytes round-tripped)\n", h.nbytes);
  return 0;
}
