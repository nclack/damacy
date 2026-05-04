#include "fixture.h"

#include "util/crc32c.h"
#include "zarr_shard_index.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef WRITE_ZARR_SCRIPT
#error "WRITE_ZARR_SCRIPT must be defined by the build system"
#endif

int
fixture_write_file(const char* path, const char* contents)
{
  FILE* file = fopen(path, "w");
  if (!file)
    return 1;
  size_t n_bytes = strlen(contents);
  size_t n_written = fwrite(contents, 1, n_bytes, file);
  fclose(file);
  return n_written == n_bytes ? 0 : 1;
}

void
fixture_write_le64(uint8_t* dst, uint64_t value)
{
  for (int i = 0; i < 8; ++i)
    dst[i] = (uint8_t)(value >> (8 * i));
}

void
fixture_write_le32(uint8_t* dst, uint32_t value)
{
  for (int i = 0; i < 4; ++i)
    dst[i] = (uint8_t)(value >> (8 * i));
}

int
fixture_write_synthetic_shard(const char* path,
                              size_t payload_n_bytes,
                              const uint64_t* offsets,
                              const uint64_t* nbytes,
                              size_t n_entries)
{
  size_t footer_n_bytes = zarr_shard_index_size(n_entries);
  size_t total_n_bytes = payload_n_bytes + footer_n_bytes;
  uint8_t* buf = (uint8_t*)calloc(total_n_bytes, 1);
  if (!buf)
    return 1;
  uint8_t* footer = buf + payload_n_bytes;
  for (size_t i = 0; i < n_entries; ++i) {
    fixture_write_le64(footer + 16 * i + 0, offsets[i]);
    fixture_write_le64(footer + 16 * i + 8, nbytes[i]);
  }
  uint32_t crc = crc32c(footer, n_entries * 16u);
  fixture_write_le32(footer + n_entries * 16u, crc);

  FILE* file = fopen(path, "wb");
  if (!file) {
    free(buf);
    return 1;
  }
  size_t n_written = fwrite(buf, 1, total_n_bytes, file);
  fclose(file);
  free(buf);
  return n_written == total_n_bytes ? 0 : 1;
}

// Append "%lld" / "%lld,%lld,..." to `dst` from `vals[0..rank)`. Returns
// the number of bytes written (excluding NUL); -1 on truncation.
static int
append_csv_i64(char* dst, size_t cap, const int64_t* vals, uint8_t rank)
{
  size_t off = 0;
  for (uint8_t d = 0; d < rank; ++d) {
    int n = snprintf(
      dst + off, cap - off, d == 0 ? "%lld" : ",%lld", (long long)vals[d]);
    if (n < 0 || (size_t)n >= cap - off)
      return -1;
    off += (size_t)n;
  }
  return (int)off;
}

int
fixture_write_zarr(const char* path,
                   const int64_t* shape,
                   const int64_t* inner,
                   const int64_t* shard,
                   uint8_t rank,
                   const char* dtype,
                   int64_t fill_offset)
{
  if (!path || !shape || !inner || !shard || !dtype || rank == 0)
    return 1;

  char shape_csv[128] = { 0 };
  char inner_csv[128] = { 0 };
  char shard_csv[128] = { 0 };
  if (append_csv_i64(shape_csv, sizeof shape_csv, shape, rank) < 0 ||
      append_csv_i64(inner_csv, sizeof inner_csv, inner, rank) < 0 ||
      append_csv_i64(shard_csv, sizeof shard_csv, shard, rank) < 0)
    return 1;

  char cmd[2048];
  int n = snprintf(cmd,
                   sizeof cmd,
                   "uv run --script %s --out %s --shape %s --inner %s "
                   "--shard %s --dtype %s --offset %lld",
                   WRITE_ZARR_SCRIPT,
                   path,
                   shape_csv,
                   inner_csv,
                   shard_csv,
                   dtype,
                   (long long)fill_offset);
  if (n < 0 || (size_t)n >= sizeof cmd)
    return 1;
  return system(cmd);
}

void
fixture_rm_tree(const char* dir)
{
  char cmd[1024];
  snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
  int rc = system(cmd);
  (void)rc;
}
