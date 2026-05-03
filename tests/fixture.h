// Shared test fixture helpers for the fs-backed integration tests
// (test_zarr_meta_cache, test_zarr_shard_cache, test_planner). Provides
// the EXPECT / RUN macros indirectly via expect.h; declares fs-side
// scaffolding implemented in fixture.c.
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
  // Returns 0 on success.
  int fixture_write_synthetic_shard(const char* path,
                                    size_t payload_n_bytes,
                                    const uint64_t* offsets,
                                    const uint64_t* nbytes,
                                    size_t n_entries);

  // Same shape as fixture_write_synthetic_shard, but the caller supplies
  // the payload bytes (typically a packed run of zstd-compressed
  // chunks). offsets/nbytes index into payload[0..payload_n_bytes).
  int fixture_write_shard_with_payload(const char* path,
                                       const void* payload,
                                       size_t payload_n_bytes,
                                       const uint64_t* offsets,
                                       const uint64_t* nbytes,
                                       size_t n_entries);

  // zstd level 3 compress src[0..src_n_bytes) into dst[0..dst_capacity);
  // writes the compressed length to *out_n_bytes. Returns 0 on success.
  int fixture_zstd_compress(const void* src,
                            size_t src_n_bytes,
                            void* dst,
                            size_t dst_capacity,
                            size_t* out_n_bytes);

  // Best-effort recursive rmdir via "rm -rf". Caller must ensure `dir`
  // is a tmpdir under their control — there is no path sanitisation.
  void fixture_rm_tree(const char* dir);

#ifdef __cplusplus
}
#endif
