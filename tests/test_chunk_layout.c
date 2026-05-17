// Integration tests for the per-array blosc1 chunk_layout probe.
//
// Two angles:
//   - direct probe against a hand-crafted file with a valid blosc1
//     header (no python; exercises just zarr_chunk_layout_probe)
//   - end-to-end through the planner with a real blosc-zstd fixture
//     written by tests/write_zarr.py — asserts the planner populates
//     sample_plan.layout / layout_probed and that non-blosc codecs
//     leave layout_probed at 0.

#include "damacy.h"
#include "fixture.h"
#include "planner/planner.h"
#include "store/store.h"
#include "util/path_intern.h"
#include "zarr/zarr_chunk_layout.h"
#include "zarr/zarr_meta_cache.h"
#include "zarr/zarr_metadata.h"
#include "zarr/zarr_shard_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define PAGE 4096u

static void
put_le32(uint8_t* p, uint32_t v)
{
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16);
  p[3] = (uint8_t)(v >> 24);
}

// Write a 16-byte blosc1-zstd header (shuffle, dont_split, compformat=4)
// to `path` followed by `payload_pad` zero bytes so the file has size
// >= header + pad. Caller picks the field values.
static int
write_blosc1_header(const char* path,
                    uint8_t typesize,
                    uint32_t nbytes,
                    uint32_t blocksize,
                    uint32_t cbytes,
                    size_t payload_pad)
{
  uint8_t buf[16] = { 0 };
  buf[0] = 2;    // version
  buf[1] = 1;    // versionlz
  buf[2] = 0x91; // flags: shuffle | dont_split | (zstd<<5)
  buf[3] = typesize;
  put_le32(buf + 4, nbytes);
  put_le32(buf + 8, blocksize);
  put_le32(buf + 12, cbytes);
  FILE* f = fopen(path, "wb");
  if (!f)
    return 1;
  if (fwrite(buf, 1, sizeof buf, f) != sizeof buf) {
    fclose(f);
    return 1;
  }
  for (size_t i = 0; i < payload_pad; ++i) {
    uint8_t z = 0;
    if (fwrite(&z, 1, 1, f) != 1) {
      fclose(f);
      return 1;
    }
  }
  fclose(f);
  return 0;
}

static int
test_probe_direct(void)
{
  char root[] = "/tmp/damacy_layout_direct_XXXXXX";
  EXPECT(mkdtemp(root));
  char path[256];
  snprintf(path, sizeof path, "%s/chunk", root);
  EXPECT(write_blosc1_header(path,
                             /*typesize*/ 2,
                             /*nbytes*/ 1024,
                             /*blocksize*/ 512,
                             /*cbytes*/ 800,
                             /*pad*/ 1024) == 0);

  struct store_fs_config sc = { .root = root, .nthreads = 1 };
  struct store* store = store_fs_create(&sc);
  EXPECT(store);

  struct chunk_layout out = { 0 };
  EXPECT(zarr_chunk_layout_probe(store,
                                 "chunk",
                                 /*off*/ 0,
                                 /*cbytes*/ 800,
                                 (uint8_t)CODEC_BLOSC_ZSTD,
                                 &out) == 0);
  EXPECT(out.codec_id == (uint8_t)CODEC_BLOSC_ZSTD);
  EXPECT(out.typesize == 2);
  EXPECT(out.blocksize == 512);
  EXPECT(out.nbytes == 1024);
  EXPECT(out.nblocks == 2);
  EXPECT(out.shuffle == 1);
  EXPECT(out.bitshuffle == 0);
  EXPECT(out.memcpyed == 0);
  EXPECT(out.dont_split == 1);

  // Non-blosc codec → probe returns non-zero, *out left as-is.
  struct chunk_layout sentinel = { .typesize = 0xAB };
  EXPECT(zarr_chunk_layout_probe(
           store, "chunk", 0, 800, (uint8_t)CODEC_ZSTD, &sentinel) != 0);
  EXPECT(sentinel.typesize == 0xAB);

  // Too-small cbytes (< 16) → fail without an IO submission.
  EXPECT(zarr_chunk_layout_probe(
           store, "chunk", 0, 8, (uint8_t)CODEC_BLOSC_ZSTD, &out) != 0);

  store_destroy(store);
  fixture_rm_tree(root);
  return 0;
}

// Malformed header: blocksize == 0 → probe fails.
static int
test_probe_rejects_bad_header(void)
{
  char root[] = "/tmp/damacy_layout_bad_XXXXXX";
  EXPECT(mkdtemp(root));
  char path[256];
  snprintf(path, sizeof path, "%s/chunk", root);
  EXPECT(write_blosc1_header(path, 2, 1024, /*blocksize*/ 0, 800, 1024) == 0);

  struct store_fs_config sc = { .root = root, .nthreads = 1 };
  struct store* store = store_fs_create(&sc);
  EXPECT(store);
  struct chunk_layout out = { 0 };
  EXPECT(zarr_chunk_layout_probe(
           store, "chunk", 0, 800, (uint8_t)CODEC_BLOSC_ZSTD, &out) != 0);
  store_destroy(store);
  fixture_rm_tree(root);
  return 0;
}

// Cache layer: set() is idempotent; get() returns NULL before set,
// non-NULL after.
static int
test_meta_cache_layout_roundtrip(void)
{
  char root[] = "/tmp/damacy_layout_cache_XXXXXX";
  EXPECT(mkdtemp(root));
  char path[256];
  snprintf(path, sizeof path, "%s/foo", root);
  EXPECT(mkdir(path, 0755) == 0);
  // Minimal sharded-zstd zarr.json so meta cache can populate the entry.
  static const char* J =
    "{\"zarr_format\":3,\"node_type\":\"array\","
    "\"shape\":[4,8],\"data_type\":\"uint16\","
    "\"chunk_grid\":{\"name\":\"regular\",\"configuration\":{"
    "\"chunk_shape\":[4,8]}},"
    "\"chunk_key_encoding\":{\"name\":\"default\",\"configuration\":{"
    "\"separator\":\"/\"}},"
    "\"fill_value\":0,"
    "\"codecs\":[{\"name\":\"sharding_indexed\",\"configuration\":{"
    "\"chunk_shape\":[2,4],"
    "\"codecs\":[{\"name\":\"bytes\",\"configuration\":{\"endian\":"
    "\"little\"}},{\"name\":\"zstd\",\"configuration\":{\"level\":3}}],"
    "\"index_codecs\":[{\"name\":\"bytes\",\"configuration\":{"
    "\"endian\":\"little\"}},{\"name\":\"crc32c\"}],"
    "\"index_location\":\"end\"}}]"
    "}";
  snprintf(path, sizeof path, "%s/foo/zarr.json", root);
  EXPECT(fixture_write_file(path, J) == 0);

  struct store_fs_config sc = { .root = root, .nthreads = 1 };
  struct store* store = store_fs_create(&sc);
  EXPECT(store);
  struct zarr_meta_cache* c = zarr_meta_cache_create(store, 4);
  EXPECT(c);
  struct zarr_metadata m = { 0 };
  EXPECT(zarr_meta_cache_get(c, "foo", &m) == DAMACY_OK);

  {
    struct chunk_layout probe = { 0 };
    EXPECT(zarr_meta_cache_layout_get(c, "foo", &probe) != 0);
  }

  struct chunk_layout cl = { .codec_id = (uint8_t)CODEC_BLOSC_ZSTD,
                             .typesize = 4,
                             .blocksize = 256,
                             .nbytes = 1024,
                             .nblocks = 4,
                             .shuffle = 1 };
  EXPECT(zarr_meta_cache_layout_set(c, "foo", &cl) == 0);
  {
    struct chunk_layout got = { 0 };
    EXPECT(zarr_meta_cache_layout_get(c, "foo", &got) == 0);
    EXPECT(got.typesize == 4);
    EXPECT(got.blocksize == 256);
    EXPECT(got.nblocks == 4);
  }

  // Second set is a no-op: first probe wins.
  struct chunk_layout cl2 = { .typesize = 1 };
  EXPECT(zarr_meta_cache_layout_set(c, "foo", &cl2) == 0);
  {
    struct chunk_layout got = { 0 };
    EXPECT(zarr_meta_cache_layout_get(c, "foo", &got) == 0);
    EXPECT(got.typesize == 4);
  }

  zarr_meta_cache_destroy(c);
  store_destroy(store);
  fixture_rm_tree(root);
  return 0;
}

// End-to-end: planner populates sample_plan.layout / layout_probed on
// the first non-fill emit for blosc-zstd arrays. Non-blosc-zstd arrays
// leave layout_probed at 0 — the probe rejects them up front.
static int
test_planner_populates_layout_blosc_zstd(void)
{
  char root[64];
  strcpy(root, "/tmp/damacy_layout_planner_XXXXXX");
  EXPECT(mkdtemp(root));
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);

  int64_t shape[2] = { 16, 32 };
  int64_t inner[2] = { 8, 16 };
  int64_t shard[2] = { 16, 32 };
  EXPECT(fixture_write_zarr_codec(
           p, shape, inner, shard, 2, "uint16", 0, "blosc-zstd") == 0);

  struct store_fs_config sc = { .root = root, .nthreads = 1 };
  struct store* store = store_fs_create(&sc);
  EXPECT(store);
  struct zarr_meta_cache* meta = zarr_meta_cache_create(store, 4);
  EXPECT(meta);
  struct zarr_shard_cache* shards = zarr_shard_cache_create(store, 4);
  EXPECT(shards);
  struct planner_config pcfg = {
    .meta_cache = meta,
    .shard_cache = shards,
    .page_alignment = PAGE,
  };
  struct planner* planner = NULL;
  EXPECT(planner_create(&pcfg, &planner) == DAMACY_OK);

  struct damacy_sample s = { .uri = "foo", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };
  int64_t dst_strides[3] = { 16 * 32, 32, 1 };

  struct read_op reads[16] = { 0 };
  struct chunk_plan chunks[16] = { 0 };
  struct sample_plan samples[4] = { 0 };
  struct path_intern paths = { 0 };
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 16,
    .chunk_plans = chunks,
    .chunk_plans_cap = 16,
    .sample_plans = samples,
    .sample_plans_cap = 4,
    .paths = &paths,
  };
  EXPECT(planner_plan(planner, &s, 1, 0, dst_strides, 3, &out) == DAMACY_OK);
  EXPECT(out.n_sample_plans == 1);
  EXPECT(samples[0].layout_probed == 1);
  EXPECT(samples[0].layout.codec_id == (uint8_t)CODEC_BLOSC_ZSTD);
  EXPECT(samples[0].layout.typesize == 2);
  EXPECT(samples[0].layout.blocksize > 0);
  EXPECT(samples[0].layout.nblocks >= 1);

  struct sample_plan samples2[4] = { 0 };
  out.sample_plans = samples2;
  EXPECT(planner_plan(planner, &s, 1, 0, dst_strides, 3, &out) == DAMACY_OK);
  EXPECT(samples2[0].layout_probed == 1);
  EXPECT(samples2[0].layout.typesize == samples[0].layout.typesize);
  EXPECT(samples2[0].layout.nblocks == samples[0].layout.nblocks);

  path_intern_free(&paths);
  planner_destroy(planner);
  zarr_shard_cache_destroy(shards);
  zarr_meta_cache_destroy(meta);
  store_destroy(store);
  fixture_rm_tree(root);
  return 0;
}

static int
test_planner_skips_layout_for_zstd(void)
{
  char root[64];
  strcpy(root, "/tmp/damacy_layout_zstd_XXXXXX");
  EXPECT(mkdtemp(root));
  char p[256];
  snprintf(p, sizeof p, "%s/foo", root);

  int64_t shape[2] = { 16, 32 };
  int64_t inner[2] = { 8, 16 };
  int64_t shard[2] = { 16, 32 };
  EXPECT(fixture_write_zarr_codec(
           p, shape, inner, shard, 2, "uint16", 0, "zstd") == 0);

  struct store_fs_config sc = { .root = root, .nthreads = 1 };
  struct store* store = store_fs_create(&sc);
  EXPECT(store);
  struct zarr_meta_cache* meta = zarr_meta_cache_create(store, 4);
  EXPECT(meta);
  struct zarr_shard_cache* shards = zarr_shard_cache_create(store, 4);
  EXPECT(shards);
  struct planner_config pcfg = {
    .meta_cache = meta,
    .shard_cache = shards,
    .page_alignment = PAGE,
  };
  struct planner* planner = NULL;
  EXPECT(planner_create(&pcfg, &planner) == DAMACY_OK);

  struct damacy_sample s = { .uri = "foo", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };
  int64_t dst_strides[3] = { 16 * 32, 32, 1 };

  struct read_op reads[16] = { 0 };
  struct chunk_plan chunks[16] = { 0 };
  struct sample_plan samples[4] = { 0 };
  struct path_intern paths = { 0 };
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 16,
    .chunk_plans = chunks,
    .chunk_plans_cap = 16,
    .sample_plans = samples,
    .sample_plans_cap = 4,
    .paths = &paths,
  };
  EXPECT(planner_plan(planner, &s, 1, 0, dst_strides, 3, &out) == DAMACY_OK);
  EXPECT(out.n_sample_plans == 1);
  // Non-blosc codecs have no blosc1 chunk header — probe returns
  // non-zero and layout_probed stays 0.
  EXPECT(samples[0].layout_probed == 0);

  path_intern_free(&paths);
  planner_destroy(planner);
  zarr_shard_cache_destroy(shards);
  zarr_meta_cache_destroy(meta);
  store_destroy(store);
  fixture_rm_tree(root);
  return 0;
}

int
main(void)
{
  RUN(test_probe_direct);
  RUN(test_probe_rejects_bad_header);
  RUN(test_meta_cache_layout_roundtrip);
  RUN(test_planner_populates_layout_blosc_zstd);
  RUN(test_planner_skips_layout_for_zstd);
  log_info("all tests passed");
  return 0;
}
