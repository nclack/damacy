// Shared test fixture helpers for the fs-backed integration tests
// (test_zarr_meta_cache, test_zarr_shard_cache, test_planner,
// test_damacy). Provides the EXPECT / RUN macros indirectly via
// expect.h; declares fs-side scaffolding implemented in fixture.c.
#pragma once

#include "expect.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Write `contents` (a NUL-terminated string) to `path`. Returns 0 on
  // success.
  int fixture_write_file(const char* path, const char* contents);

  // Pack a u64 / u32 little-endian into `dst` (8 / 4 bytes respectively).
  void fixture_write_le64(uint8_t* dst, uint64_t value);
  void fixture_write_le32(uint8_t* dst, uint32_t value);

  // Build a synthetic zarr-v3 sharded chunk file at `path`:
  // `payload_n_bytes` of zeroed payload followed by an index footer of
  // `n_entries` (offset_u64, nbytes_u64) pairs and a CRC32C terminator.
  // Used by the cache + planner tests where mechanic-level control over
  // the offset table matters more than real chunk content.
  int fixture_write_synthetic_shard(const char* path,
                                    size_t payload_n_bytes,
                                    const uint64_t* offsets,
                                    const uint64_t* nbytes,
                                    size_t n_entries);

  // Shell out to tests/write_zarr.py (uv-script) to produce a real
  // sharded zstd zarr-v3 array at `path`. Content is a deterministic
  // row-major linearization plus `fill_offset`, masked to the dtype's
  // range. dtype is a zarr-v3 string ("uint8", "uint16", ...). Returns 0
  // on success.
  int fixture_write_zarr(const char* path,
                         const int64_t* shape,
                         const int64_t* inner,
                         const int64_t* shard,
                         uint8_t rank,
                         const char* dtype,
                         int64_t fill_offset);

  // Same, but with a configurable inner codec. `codec` is one of "zstd",
  // "blosc-zstd" (matching tests/write_zarr.py --codec).
  int fixture_write_zarr_codec(const char* path,
                               const int64_t* shape,
                               const int64_t* inner,
                               const int64_t* shard,
                               uint8_t rank,
                               const char* dtype,
                               int64_t fill_offset,
                               const char* codec);

  // Best-effort recursive rmdir via "rm -rf". Caller must ensure `dir`
  // is a tmpdir under their control — there is no path sanitisation.
  void fixture_rm_tree(const char* dir);

#ifdef __cplusplus
}
#endif
