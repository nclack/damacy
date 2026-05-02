#include "zarr_shard_index.h"

#include "util/crc32c.h"

#include <string.h>

static uint64_t
load_le_u64(const uint8_t* p)
{
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i)
    v |= ((uint64_t)p[i]) << (8 * i);
  return v;
}

static uint32_t
load_le_u32(const uint8_t* p)
{
  return ((uint32_t)p[0]) | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

int
zarr_shard_index_parse(const void* buf,
                       size_t buf_len,
                       size_t n_entries,
                       struct zarr_shard_entry* entries)
{
  if (!buf || !entries)
    return 1;
  size_t want = zarr_shard_index_size(n_entries);
  if (buf_len != want)
    return 1;

  const uint8_t* p = (const uint8_t*)buf;
  // Verify CRC32C over the [offset,nbytes] table.
  uint32_t got = load_le_u32(p + n_entries * 16u);
  uint32_t expected = crc32c(p, n_entries * 16u);
  if (got != expected)
    return 1;

  for (size_t i = 0; i < n_entries; ++i) {
    entries[i].offset = load_le_u64(p + i * 16u);
    entries[i].nbytes = load_le_u64(p + i * 16u + 8u);
  }
  return 0;
}
