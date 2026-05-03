// Integration test for zarr_shard_cache. Builds a tmpdir-backed fs
// store, writes a synthetic shard file whose footer is a valid
// (offset, nbytes) index protected by CRC32C, and exercises the cache.

#include "store.h"
#include "util/crc32c.h"
#include "zarr_metadata.h"
#include "zarr_shard_cache.h"
#include "zarr_shard_index.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define EXPECT(cond)                                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(                                                                 \
        stderr, "  expect failed at %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static void
write_le64(uint8_t* dst, uint64_t v)
{
  for (int i = 0; i < 8; ++i)
    dst[i] = (uint8_t)(v >> (8 * i));
}

static void
write_le32(uint8_t* dst, uint32_t v)
{
  for (int i = 0; i < 4; ++i)
    dst[i] = (uint8_t)(v >> (8 * i));
}

// Build a synthetic shard file at `path`: PAYLOAD_BYTES of dummy data
// followed by a valid index footer for `n_entries` entries with the
// supplied offsets/sizes.
static int
write_synthetic_shard(const char* path,
                      size_t payload_bytes,
                      const uint64_t* offsets,
                      const uint64_t* nbytes,
                      size_t n_entries)
{
  size_t idx_bytes = zarr_shard_index_size(n_entries);
  size_t total = payload_bytes + idx_bytes;
  uint8_t* buf = (uint8_t*)calloc(total, 1);
  if (!buf)
    return 1;
  uint8_t* footer = buf + payload_bytes;
  for (size_t i = 0; i < n_entries; ++i) {
    write_le64(footer + 16 * i + 0, offsets[i]);
    write_le64(footer + 16 * i + 8, nbytes[i]);
  }
  uint32_t c = crc32c(footer, n_entries * 16u);
  write_le32(footer + n_entries * 16u, c);

  FILE* f = fopen(path, "wb");
  if (!f) {
    free(buf);
    return 1;
  }
  size_t w = fwrite(buf, 1, total, f);
  fclose(f);
  free(buf);
  return w == total ? 0 : 1;
}

static void
rm_tree(const char* dir)
{
  char cmd[1024];
  snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
  int rc = system(cmd);
  (void)rc;
}

static int
test_shard_cache(void)
{
  char tmpl[] = "/tmp/damacy_shard_cache_XXXXXX";
  char* root = mkdtemp(tmpl);
  EXPECT(root);

  // Synthetic 2D array: shard_shape = [4, 4], inner = [2, 2], so
  // n_inner_per_shard = 4. Two shards: c/0/0 and c/0/1.
  const uint64_t offsets[4] = { 0, 100, 200, 300 };
  const uint64_t nbytes[4] = { 50, 50, 50, 50 };

  char path[512];
  snprintf(path, sizeof path, "%s/foo", root);
  EXPECT(mkdir(path, 0755) == 0);
  snprintf(path, sizeof path, "%s/foo/c", root);
  EXPECT(mkdir(path, 0755) == 0);
  snprintf(path, sizeof path, "%s/foo/c/0", root);
  EXPECT(mkdir(path, 0755) == 0);
  snprintf(path, sizeof path, "%s/foo/c/0/0", root);
  EXPECT(write_synthetic_shard(path, 4096, offsets, nbytes, 4) == 0);
  snprintf(path, sizeof path, "%s/foo/c/0/1", root);
  EXPECT(write_synthetic_shard(path, 4096, offsets, nbytes, 4) == 0);

  struct store_fs_config sc = { .root = root, .nthreads = 1 };
  struct store* store = store_fs_create(&sc);
  EXPECT(store);

  // Hand-build the metadata describing the shard layout (we don't need
  // the meta cache here; the shard cache takes meta as a parameter).
  struct zarr_metadata meta = {
    .rank = 2,
    .shape = { 4, 8 },
    .inner_chunk_shape = { 2, 2 },
    .shard_shape = { 4, 4 },
    .sharded = 1,
    .index_location_end = 1,
  };

  struct zarr_shard_cache* c = zarr_shard_cache_create(store, 4);
  EXPECT(c);

  const uint64_t coord00[2] = { 0, 0 };
  const struct zarr_shard_entry* entries = NULL;
  uint64_t n = 0;

  // Miss → load → return.
  EXPECT(zarr_shard_cache_get(c, "foo", &meta, coord00, &entries, &n) ==
         DAMACY_OK);
  EXPECT(n == 4);
  EXPECT(entries != NULL);
  EXPECT(entries[0].offset == 0 && entries[0].nbytes == 50);
  EXPECT(entries[3].offset == 300 && entries[3].nbytes == 50);

  // Same coord → hit.
  const struct zarr_shard_entry* entries_b = NULL;
  uint64_t nb = 0;
  EXPECT(zarr_shard_cache_get(c, "foo", &meta, coord00, &entries_b, &nb) ==
         DAMACY_OK);
  EXPECT(entries_b == entries);

  struct zarr_shard_cache_stats st;
  zarr_shard_cache_stats_get(c, &st);
  EXPECT(st.hits == 1);
  EXPECT(st.misses == 1);
  EXPECT(st.size == 1);

  // Different shard coord → another miss; both cached.
  const uint64_t coord01[2] = { 0, 1 };
  EXPECT(zarr_shard_cache_get(c, "foo", &meta, coord01, &entries_b, &nb) ==
         DAMACY_OK);
  EXPECT(entries_b != entries);
  zarr_shard_cache_stats_get(c, &st);
  EXPECT(st.size == 2);

  // Missing shard → NOTFOUND.
  const uint64_t coordbad[2] = { 99, 99 };
  EXPECT(zarr_shard_cache_get(c, "foo", &meta, coordbad, &entries_b, &nb) ==
         DAMACY_NOTFOUND);

  zarr_shard_cache_destroy(c);
  store_destroy(store);
  rm_tree(root);
  return 0;
}

#define RUN(t)                                                                 \
  do {                                                                         \
    int r = t();                                                               \
    if (r != 0) {                                                              \
      fprintf(stderr, "FAIL %s\n", #t);                                        \
      return r;                                                                \
    }                                                                          \
    printf("ok   %s\n", #t);                                                   \
  } while (0)

int
main(void)
{
  RUN(test_shard_cache);
  printf("all tests passed\n");
  return 0;
}
