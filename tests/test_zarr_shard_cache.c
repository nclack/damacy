// Integration test for zarr_shard_cache. Builds a tmpdir-backed fs
// store, writes a synthetic shard file whose footer is a valid
// (offset, nbytes) index protected by CRC32C, and exercises the cache.
//
// Hit paths below pass raw "foo" literals and create the cache with
// uris=NULL; correctness relies on the compiler pooling equal literals
// within a TU. The pointer-identity test exercises path_intern_acquire
// explicitly.

#include "fixture.h"
#include "store/store.h"
#include "util/hash.h"
#include "util/path_intern.h"
#include "zarr/zarr_metadata.h"
#include "zarr/zarr_shard_cache.h"
#include "zarr/zarr_shard_index.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

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
  EXPECT(fixture_write_synthetic_shard(path, 4096, offsets, nbytes, 4) == 0);
  snprintf(path, sizeof path, "%s/foo/c/0/1", root);
  EXPECT(fixture_write_synthetic_shard(path, 4096, offsets, nbytes, 4) == 0);

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

  struct zarr_shard_cache* c = zarr_shard_cache_create(store, NULL, 4);
  EXPECT(c);

  const uint64_t coord00[2] = { 0, 0 };
  const struct zarr_shard_entry* entries = NULL;
  uint64_t n = 0;
  struct zarr_shard_pin pin = { 0 };

  // Miss → load → return.
  EXPECT(
    zarr_shard_cache_get(
      c, "foo", hash_fnv1a_str("foo"), &meta, coord00, &pin, &entries, &n) ==
    DAMACY_OK);
  EXPECT(n == 4);
  EXPECT(entries != NULL);
  EXPECT(entries[0].offset == 0 && entries[0].nbytes == 50);
  EXPECT(entries[3].offset == 300 && entries[3].nbytes == 50);

  // Same coord → hit.
  const struct zarr_shard_entry* entries_b = NULL;
  uint64_t nb = 0;
  struct zarr_shard_pin pin_b = { 0 };
  EXPECT(zarr_shard_cache_get(c,
                              "foo",
                              hash_fnv1a_str("foo"),
                              &meta,
                              coord00,
                              &pin_b,
                              &entries_b,
                              &nb) == DAMACY_OK);
  EXPECT(entries_b == entries);

  struct zarr_shard_cache_stats st;
  zarr_shard_cache_stats_get(c, &st);
  EXPECT(st.counters.hits == 1);
  EXPECT(st.counters.misses == 1);
  EXPECT(st.size == 1);
  // Pins are per-acquisition: two pins to the same entry refcount 2 -> 1 -> 0.
  zarr_shard_cache_release(c, pin_b);
  zarr_shard_cache_release(c, pin);

  // Different shard coord → another miss; both cached.
  const uint64_t coord01[2] = { 0, 1 };
  EXPECT(zarr_shard_cache_get(c,
                              "foo",
                              hash_fnv1a_str("foo"),
                              &meta,
                              coord01,
                              &pin_b,
                              &entries_b,
                              &nb) == DAMACY_OK);
  EXPECT(entries_b != entries);
  zarr_shard_cache_stats_get(c, &st);
  EXPECT(st.size == 2);
  zarr_shard_cache_release(c, pin_b);

  // Missing shard → NOTFOUND. pin must remain unpinned on non-OK.
  const uint64_t coordbad[2] = { 99, 99 };
  struct zarr_shard_pin pin_bad = { (void*)0xdeadbeef };
  EXPECT(zarr_shard_cache_get(c,
                              "foo",
                              hash_fnv1a_str("foo"),
                              &meta,
                              coordbad,
                              &pin_bad,
                              &entries_b,
                              &nb) == DAMACY_NOTFOUND);
  EXPECT(pin_bad.opaque == NULL);

  zarr_shard_cache_destroy(c);
  store_destroy(store);
  fixture_rm_tree(root);
  return 0;
}

// index_location: "start". The index sits at offset 0 with absolute
// offsets pointing into the body that follows.
static int
test_shard_cache_index_start(void)
{
  char tmpl[] = "/tmp/damacy_shard_cache_start_XXXXXX";
  char* root = mkdtemp(tmpl);
  EXPECT(root);

  // 4 inner chunks per shard; index sits at offset 0.
  // index_size = 4*16 + 4 = 68. Pad data offsets clear of the header.
  const uint64_t offsets[4] = { 128, 256, 384, 512 };
  const uint64_t nbytes[4] = { 64, 64, 64, 64 };

  char path[512];
  snprintf(path, sizeof path, "%s/foo", root);
  EXPECT(mkdir(path, 0755) == 0);
  snprintf(path, sizeof path, "%s/foo/c", root);
  EXPECT(mkdir(path, 0755) == 0);
  snprintf(path, sizeof path, "%s/foo/c/0", root);
  EXPECT(mkdir(path, 0755) == 0);
  snprintf(path, sizeof path, "%s/foo/c/0/0", root);
  EXPECT(fixture_write_synthetic_shard_start(
           path, /*payload=*/2048, offsets, nbytes, 4) == 0);

  struct store_fs_config sc = { .root = root, .nthreads = 1 };
  struct store* store = store_fs_create(&sc);
  EXPECT(store);

  struct zarr_metadata meta = {
    .rank = 2,
    .shape = { 4, 4 },
    .inner_chunk_shape = { 2, 2 },
    .shard_shape = { 4, 4 },
    .sharded = 1,
    .index_location_end = 0,
  };

  struct zarr_shard_cache* c = zarr_shard_cache_create(store, NULL, 4);
  EXPECT(c);

  const uint64_t coord00[2] = { 0, 0 };
  const struct zarr_shard_entry* entries = NULL;
  uint64_t n = 0;
  struct zarr_shard_pin pin = { 0 };
  EXPECT(
    zarr_shard_cache_get(
      c, "foo", hash_fnv1a_str("foo"), &meta, coord00, &pin, &entries, &n) ==
    DAMACY_OK);
  EXPECT(n == 4);
  EXPECT(entries[0].offset == 128 && entries[0].nbytes == 64);
  EXPECT(entries[3].offset == 512 && entries[3].nbytes == 64);
  zarr_shard_cache_release(c, pin);

  zarr_shard_cache_destroy(c);
  store_destroy(store);
  fixture_rm_tree(root);
  return 0;
}

// Non-sharded array: each "shard" is a 1-entry synthetic index built
// from the file's size, no on-disk footer.
static int
test_shard_cache_unsharded(void)
{
  char tmpl[] = "/tmp/damacy_shard_cache_unsharded_XXXXXX";
  char* root = mkdtemp(tmpl);
  EXPECT(root);

  // Layout: c/0/0 and c/1/2 (just two distinct chunk files).
  char path[512];
  snprintf(path, sizeof path, "%s/foo", root);
  EXPECT(mkdir(path, 0755) == 0);
  snprintf(path, sizeof path, "%s/foo/c", root);
  EXPECT(mkdir(path, 0755) == 0);
  snprintf(path, sizeof path, "%s/foo/c/0", root);
  EXPECT(mkdir(path, 0755) == 0);
  snprintf(path, sizeof path, "%s/foo/c/1", root);
  EXPECT(mkdir(path, 0755) == 0);
  snprintf(path, sizeof path, "%s/foo/c/0/0", root);
  EXPECT(fixture_write_zero_file(path, 123) == 0);
  snprintf(path, sizeof path, "%s/foo/c/1/2", root);
  EXPECT(fixture_write_zero_file(path, 456) == 0);

  struct store_fs_config sc = { .root = root, .nthreads = 1 };
  struct store* store = store_fs_create(&sc);
  EXPECT(store);

  // Non-sharded: inner == shard. n_inner_per_shard = 1.
  struct zarr_metadata meta = {
    .rank = 2,
    .shape = { 4, 8 },
    .inner_chunk_shape = { 2, 4 },
    .shard_shape = { 2, 4 },
    .sharded = 0,
    .index_location_end = 1,
  };

  struct zarr_shard_cache* c = zarr_shard_cache_create(store, NULL, 4);
  EXPECT(c);

  const uint64_t coord00[2] = { 0, 0 };
  const struct zarr_shard_entry* entries = NULL;
  uint64_t n = 0;
  struct zarr_shard_pin pin = { 0 };
  EXPECT(
    zarr_shard_cache_get(
      c, "foo", hash_fnv1a_str("foo"), &meta, coord00, &pin, &entries, &n) ==
    DAMACY_OK);
  EXPECT(n == 1);
  EXPECT(entries[0].offset == 0);
  EXPECT(entries[0].nbytes == 123);
  zarr_shard_cache_release(c, pin);

  const uint64_t coord12[2] = { 1, 2 };
  EXPECT(
    zarr_shard_cache_get(
      c, "foo", hash_fnv1a_str("foo"), &meta, coord12, &pin, &entries, &n) ==
    DAMACY_OK);
  EXPECT(n == 1);
  EXPECT(entries[0].offset == 0);
  EXPECT(entries[0].nbytes == 456);
  zarr_shard_cache_release(c, pin);

  // Missing chunk file → NOTFOUND.
  const uint64_t coordbad[2] = { 9, 9 };
  EXPECT(
    zarr_shard_cache_get(
      c, "foo", hash_fnv1a_str("foo"), &meta, coordbad, &pin, &entries, &n) ==
    DAMACY_NOTFOUND);

  zarr_shard_cache_destroy(c);
  store_destroy(store);
  fixture_rm_tree(root);
  return 0;
}

// Same shape as the meta-cache pointer-identity test: drive the shard
// cache through a real path_intern and verify hit-by-pointer-identity.
static int
test_shard_cache_pointer_identity(void)
{
  char tmpl[] = "/tmp/damacy_shard_pid_XXXXXX";
  char* root = mkdtemp(tmpl);
  EXPECT(root);

  char path[512];
  snprintf(path, sizeof path, "%s/foo", root);
  EXPECT(mkdir(path, 0755) == 0);
  snprintf(path, sizeof path, "%s/foo/c", root);
  EXPECT(mkdir(path, 0755) == 0);
  snprintf(path, sizeof path, "%s/foo/c/0", root);
  EXPECT(mkdir(path, 0755) == 0);
  snprintf(path, sizeof path, "%s/foo/c/0/0", root);
  EXPECT(fixture_write_zero_file(path, 256) == 0);
  snprintf(path, sizeof path, "%s/bar", root);
  EXPECT(mkdir(path, 0755) == 0);
  snprintf(path, sizeof path, "%s/bar/c", root);
  EXPECT(mkdir(path, 0755) == 0);
  snprintf(path, sizeof path, "%s/bar/c/0", root);
  EXPECT(mkdir(path, 0755) == 0);
  snprintf(path, sizeof path, "%s/bar/c/0/0", root);
  EXPECT(fixture_write_zero_file(path, 256) == 0);

  struct store_fs_config sc = { .root = root, .nthreads = 1 };
  struct store* store = store_fs_create(&sc);
  EXPECT(store);

  struct zarr_metadata meta = {
    .rank = 2,
    .shape = { 4, 8 },
    .inner_chunk_shape = { 2, 4 },
    .shard_shape = { 2, 4 },
    .sharded = 0,
    .index_location_end = 1,
  };

  struct path_intern uris = { 0 };
  struct zarr_shard_cache* c = zarr_shard_cache_create(store, &uris, 4);
  EXPECT(c);

  const char* foo_a = path_intern_acquire(&uris, "foo");
  const char* foo_b = path_intern_acquire(&uris, "foo");
  const char* bar = path_intern_acquire(&uris, "bar");
  EXPECT(foo_a && foo_b && bar);
  EXPECT(foo_a == foo_b);
  EXPECT(foo_a != bar);

  const uint64_t coord00[2] = { 0, 0 };
  const struct zarr_shard_entry* entries = NULL;
  uint64_t n = 0;
  struct zarr_shard_pin pin_a = { 0 };
  struct zarr_shard_pin pin_b = { 0 };
  struct zarr_shard_pin pin_bar = { 0 };

  EXPECT(zarr_shard_cache_get(c,
                              foo_a,
                              path_intern_hash(foo_a),
                              &meta,
                              coord00,
                              &pin_a,
                              &entries,
                              &n) == DAMACY_OK);
  struct zarr_shard_cache_stats st;
  zarr_shard_cache_stats_get(c, &st);
  EXPECT(st.counters.misses == 1);
  EXPECT(st.size == 1);

  EXPECT(zarr_shard_cache_get(c,
                              foo_b,
                              path_intern_hash(foo_b),
                              &meta,
                              coord00,
                              &pin_b,
                              &entries,
                              &n) == DAMACY_OK);
  zarr_shard_cache_stats_get(c, &st);
  EXPECT(st.counters.hits == 1);
  EXPECT(st.counters.misses == 1);
  EXPECT(st.size == 1);

  EXPECT(
    zarr_shard_cache_get(
      c, bar, path_intern_hash(bar), &meta, coord00, &pin_bar, &entries, &n) ==
    DAMACY_OK);
  zarr_shard_cache_stats_get(c, &st);
  EXPECT(st.counters.misses == 2);
  EXPECT(st.size == 2);

  zarr_shard_cache_release(c, pin_a);
  zarr_shard_cache_release(c, pin_b);
  zarr_shard_cache_release(c, pin_bar);
  path_intern_release(&uris, foo_a);
  path_intern_release(&uris, foo_b);
  path_intern_release(&uris, bar);
  zarr_shard_cache_destroy(c);
  path_intern_free(&uris);
  store_destroy(store);
  fixture_rm_tree(root);
  return 0;
}

int
main(void)
{
  RUN(test_shard_cache);
  RUN(test_shard_cache_index_start);
  RUN(test_shard_cache_unsharded);
  RUN(test_shard_cache_pointer_identity);
  log_info("all tests passed");
  return 0;
}
