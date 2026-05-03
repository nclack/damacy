// End-to-end smoke test for damacy: build a tiny on-disk zarr v3 store
// with real zstd-compressed inner chunks, push a sample, pop the
// assembled batch off the GPU, and verify it byte-for-byte against the
// data we wrote.
//
// Reuses the test_planner JSON (2D shape=[4,8], inner=[2,4],
// shard=[4,8], one shard, 4 inner chunks). Builds the shard with
// libzstd-compressed payloads instead of zero-filled stubs.

#include "damacy.h"
#include "fixture.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

static const char* MINIMAL_ZARR_JSON =
  "{"
  "\"zarr_format\":3,"
  "\"node_type\":\"array\","
  "\"shape\":[4,8],"
  "\"data_type\":\"uint16\","
  "\"chunk_grid\":{\"name\":\"regular\",\"configuration\":{"
  "\"chunk_shape\":[4,8]}},"
  "\"chunk_key_encoding\":{\"name\":\"default\",\"configuration\":{"
  "\"separator\":\"/\"}},"
  "\"fill_value\":0,"
  "\"codecs\":[{\"name\":\"sharding_indexed\",\"configuration\":{"
  "\"chunk_shape\":[2,4],"
  "\"codecs\":[{\"name\":\"bytes\",\"configuration\":{\"endian\":\"little\"}},"
  "{\"name\":\"zstd\",\"configuration\":{\"level\":3,\"checksum\":false}}],"
  "\"index_codecs\":[{\"name\":\"bytes\",\"configuration\":{"
  "\"endian\":\"little\"}},{\"name\":\"crc32c\"}],"
  "\"index_location\":\"end\"}}]"
  "}";

struct fixture
{
  char root[64];
  uint16_t source[4][8];
};

// Build a sharded shard file at <root>/foo/c/0/0 with each of the 4
// inner chunks zstd-compressed and packed sequentially in the payload.
// Inner chunk shape [2,4] over uint16 = 16 bytes uncompressed.
static int
build_shard(const struct fixture* f, const char* path)
{
  enum
  {
    PAYLOAD_CAP = 4096,
    N_ENTRIES = 4
  };
  uint8_t payload[PAYLOAD_CAP];
  uint64_t offsets[N_ENTRIES] = { 0 };
  uint64_t nbytes[N_ENTRIES] = { 0 };
  uint64_t cursor = 0;

  for (int chunk_y = 0; chunk_y < 2; ++chunk_y) {
    for (int chunk_x = 0; chunk_x < 2; ++chunk_x) {
      int idx = chunk_y * 2 + chunk_x;
      uint16_t chunk_buf[8];
      for (int iy = 0; iy < 2; ++iy)
        for (int ix = 0; ix < 4; ++ix)
          chunk_buf[iy * 4 + ix] =
            f->source[chunk_y * 2 + iy][chunk_x * 4 + ix];
      size_t comp_n = 0;
      if (fixture_zstd_compress(chunk_buf,
                                sizeof chunk_buf,
                                payload + cursor,
                                PAYLOAD_CAP - cursor,
                                &comp_n))
        return 1;
      offsets[idx] = cursor;
      nbytes[idx] = comp_n;
      cursor += comp_n;
    }
  }
  return fixture_write_shard_with_payload(
    path, payload, cursor, offsets, nbytes, N_ENTRIES);
}

static int
fixture_init(struct fixture* f)
{
  strcpy(f->root, "/tmp/damacy_smoke_XXXXXX");
  EXPECT(mkdtemp(f->root));
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 8; ++x)
      f->source[y][x] = (uint16_t)(y * 8 + x);

  char p[256];
  snprintf(p, sizeof p, "%s/foo", f->root);
  EXPECT(mkdir(p, 0755) == 0);
  snprintf(p, sizeof p, "%s/foo/zarr.json", f->root);
  EXPECT(fixture_write_file(p, MINIMAL_ZARR_JSON) == 0);
  snprintf(p, sizeof p, "%s/foo/c", f->root);
  EXPECT(mkdir(p, 0755) == 0);
  snprintf(p, sizeof p, "%s/foo/c/0", f->root);
  EXPECT(mkdir(p, 0755) == 0);
  snprintf(p, sizeof p, "%s/foo/c/0/0", f->root);
  EXPECT(build_shard(f, p) == 0);
  return 0;
}

static struct damacy_config
mk_cfg(const char* root)
{
  return (struct damacy_config){
    .store_root = root,
    .batch_size = 1,
    .lookahead_batches = 2,
    .n_io_threads = 1,
    .host_buffer_bytes = 64ull << 10,
    .device_buffer_bytes = 64ull << 10,
    .n_zarrs_meta_cache = 4,
    .n_shards_meta_cache = 4,
    .dtype = DAMACY_U16,
  };
}

static struct damacy_sample
mk_sample(int64_t y0, int64_t y1, int64_t x0, int64_t x1)
{
  struct damacy_sample s = { .uri = "foo", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = y0, .end = y1 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = x0, .end = x1 };
  return s;
}

// Push one sample, pop the resulting batch, copy its device buffer to
// host. Caller-owned `out` must hold at least win_y * win_x u16s.
static int
run_one(struct damacy* d,
        struct damacy_sample s,
        uint16_t* out,
        size_t out_capacity_elements)
{
  struct damacy_sample_slice slice = { .beg = &s, .end = &s + 1 };
  struct damacy_push_result pr = damacy_push(d, slice);
  EXPECT(pr.status == DAMACY_OK);

  struct damacy_batch* b = NULL;
  EXPECT(damacy_pop(d, &b) == DAMACY_OK);

  struct damacy_batch_info info;
  damacy_batch_info(b, &info);
  EXPECT(info.rank == 3);
  EXPECT(info.dtype == DAMACY_U16);
  EXPECT(info.shape[0] == 1);
  size_t n_elements = (size_t)info.shape[1] * (size_t)info.shape[2];
  EXPECT(n_elements <= out_capacity_elements);
  EXPECT(cudaMemcpy(out,
                    info.device_ptr,
                    n_elements * sizeof(uint16_t),
                    cudaMemcpyDeviceToHost) == cudaSuccess);
  damacy_release(d, b);
  return 0;
}

// Sample covers the full array. Output should match `source` exactly.
static int
test_full_array(void)
{
  struct fixture f;
  if (fixture_init(&f))
    return 1;
  struct damacy_config cfg = mk_cfg(f.root);
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  uint16_t out[4 * 8] = { 0 };
  if (run_one(d, mk_sample(0, 4, 0, 8), out, 4 * 8))
    return 1;
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 8; ++x)
      EXPECT(out[y * 8 + x] == f.source[y][x]);

  damacy_destroy(d);
  fixture_rm_tree(f.root);
  return 0;
}

// Sample covers a sub-region that crosses all four chunks. The
// assemble kernel must copy the right intersection from each chunk
// into the right position of the (smaller) output.
static int
test_partial_crossing_chunks(void)
{
  struct fixture f;
  if (fixture_init(&f))
    return 1;
  struct damacy_config cfg = mk_cfg(f.root);
  struct damacy* d = NULL;
  EXPECT(damacy_create(&cfg, &d) == DAMACY_OK);

  // y[1,3) x[2,7) — touches all 4 chunks (2x2 grid). Output shape [2,5].
  uint16_t out[2 * 5] = { 0 };
  if (run_one(d, mk_sample(1, 3, 2, 7), out, 2 * 5))
    return 1;
  for (int y = 0; y < 2; ++y)
    for (int x = 0; x < 5; ++x)
      EXPECT(out[y * 5 + x] == f.source[1 + y][2 + x]);

  damacy_destroy(d);
  fixture_rm_tree(f.root);
  return 0;
}

int
main(void)
{
  RUN(test_full_array);
  RUN(test_partial_crossing_chunks);
  log_info("all tests passed");
  return 0;
}
