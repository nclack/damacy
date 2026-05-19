// Regression for the gds_submit_dev partial-submit UAF: a bogus key
// mid-batch must drain the stream and release prior pins. Runs under
// cuFile compat mode when nvidia-fs is absent.

#include "cuda_init.h"
#include "expect.h"
#include "store/store.h"
#include "store/store_fs_gds.h"
#include "util/lru.h"

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int
write_byte(const char* path, char c)
{
  FILE* f = fopen(path, "wb");
  if (!f)
    return 1;
  size_t n = fwrite(&c, 1, 1, f);
  fclose(f);
  return n == 1 ? 0 : 1;
}

static int
test_submit_fail_releases_pins(void)
{
  if (cuda_init_primary()) {
    log_info("test_store_fs_gds: no CUDA device; skipping");
    return 0;
  }

  char root[] = "/tmp/damacy_store_fs_gds_XXXXXX";
  EXPECT(mkdtemp(root));
  for (int i = 0; i < 3; ++i) {
    char p[256];
    snprintf(p, sizeof p, "%s/k%d", root, i);
    EXPECT(write_byte(p, (char)('a' + i)) == 0);
  }

  struct store_fs_gds_config cfg = {
    .root = root,
    .fd_cache_capacity = 8,
  };
  struct store* s = store_fs_gds_create(&cfg);
  if (!s) {
    log_info(
      "test_store_fs_gds: store_fs_gds_create returned NULL (no cuFile); "
      "skipping");
    return 0;
  }

  CUstream stream = NULL;
  EXPECT(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS);
  store_fs_gds_set_stream(s, stream);

  CUdeviceptr dbuf = 0;
  EXPECT(cuMemAlloc(&dbuf, 4096) == CUDA_SUCCESS);

  struct store_read reads[] = {
    { .key = "k0", .dst = (void*)dbuf, .offset = 0, .len = 1 },
    { .key = "k1", .dst = (void*)(dbuf + 16), .offset = 0, .len = 1 },
    { .key = "missing_no_such_file",
      .dst = (void*)(dbuf + 32),
      .offset = 0,
      .len = 1 },
    { .key = "k2", .dst = (void*)(dbuf + 48), .offset = 0, .len = 1 },
  };
  struct store_event ev = store_read_submit_dev(s, reads, 4);
  EXPECT(ev.seq == 0);

  struct lru_stats stats;
  store_fs_gds_stats_get(s, &stats);
  EXPECT(stats.pinned == 0);

  struct store_read good[] = {
    { .key = "k0", .dst = (void*)dbuf, .offset = 0, .len = 1 },
  };
  struct store_event ev2 = store_read_submit_dev(s, good, 1);
  EXPECT(ev2.seq != 0);
  EXPECT(cuStreamSynchronize(stream) == CUDA_SUCCESS);

  cuMemFree(dbuf);
  store_destroy(s);
  cuStreamDestroy(stream);
  return 0;
}

int
main(void)
{
  RUN(test_submit_fail_releases_pins);
  return 0;
}
