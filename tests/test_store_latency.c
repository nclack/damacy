#include "expect.h"
#include "fixture.h"
#include "store/store.h"
#include "store/store_fs.h"
#include "store/store_latency.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

static uint64_t
now_ns(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int
make_root(char* root, size_t root_cap)
{
  snprintf(root, root_cap, "/tmp/damacy_store_latency_XXXXXX");
  return mkdtemp(root) ? 0 : 1;
}

static int
write_payload(const char* root)
{
  char path[256];
  snprintf(path, sizeof path, "%s/payload", root);
  return fixture_write_zero_file(path, 4096);
}

static int
test_submit_map_and_stat_pay_latency_floor(void)
{
  char root[64];
  EXPECT(make_root(root, sizeof root) == 0);
  EXPECT(write_payload(root) == 0);

  struct store* inner =
    store_fs_create(&(struct store_fs_config){ .root = root, .nthreads = 1 });
  EXPECT(inner);

  const uint64_t floor_ns = 2ull * 1000ull * 1000ull;
  struct store* latency = store_latency_create(
    inner, &(struct damacy_latency_model){ .baseline_ns = floor_ns });
  EXPECT(latency);

  uint8_t buf[32];
  struct store_read read = {
    .key = "payload",
    .dst = buf,
    .offset = 0,
    .len = sizeof buf,
  };
  uint64_t t0 = now_ns();
  EXPECT(store_read_many(latency, &read, 1) == 0);
  EXPECT(now_ns() - t0 >= floor_ns);

  uint64_t size = 0;
  t0 = now_ns();
  EXPECT(store_stat(latency, "payload", &size) == STORE_STAT_OK);
  EXPECT(size == 4096);
  EXPECT(now_ns() - t0 >= floor_ns);

  struct store_view view = { 0 };
  t0 = now_ns();
  EXPECT(store_map(latency, "payload", &view) == 0);
  EXPECT(now_ns() - t0 >= floor_ns);
  store_unmap(latency, &view);

  struct store_latency_stats st;
  store_latency_stats_get(latency, &st);
  EXPECT(st.ops == 3);
  EXPECT(st.submit_ops == 1);
  EXPECT(st.stat_ops == 1);
  EXPECT(st.map_ops == 1);
  EXPECT(st.submit_dev_ops == 0);
  EXPECT(st.active == 0);
  EXPECT(st.max_active >= 1);
  EXPECT(st.total_sleep_ns >= 3 * floor_ns);
  EXPECT(st.max_sleep_ns >= floor_ns);

  store_latency_stats_reset(latency);
  store_latency_stats_get(latency, &st);
  EXPECT(st.ops == 0);
  EXPECT(st.submit_ops == 0);
  EXPECT(st.stat_ops == 0);
  EXPECT(st.map_ops == 0);
  EXPECT(st.active == 0);

  store_destroy(latency);
  store_destroy(inner);
  fixture_rm_tree(root);
  return 0;
}

static int
test_lognormal_samples_vary(void)
{
  char root[64];
  EXPECT(make_root(root, sizeof root) == 0);
  EXPECT(write_payload(root) == 0);

  struct store* inner =
    store_fs_create(&(struct store_fs_config){ .root = root, .nthreads = 1 });
  EXPECT(inner);
  struct store* latency =
    store_latency_create(inner,
                         &(struct damacy_latency_model){
                           .lognormal_mu_ln_ns = 13.0,
                           .lognormal_sigma_ln_ns = 1.0,
                           .cap_ns = 50ull * 1000ull * 1000ull,
                           .seed = 1,
                         });
  EXPECT(latency);

  uint8_t buf[32];
  struct store_read read = {
    .key = "payload",
    .dst = buf,
    .offset = 0,
    .len = sizeof buf,
  };
  uint64_t samples[8];
  for (int i = 0; i < 8; ++i) {
    uint64_t t0 = now_ns();
    EXPECT(store_read_many(latency, &read, 1) == 0);
    samples[i] = now_ns() - t0;
  }
  int distinct = 0;
  for (int i = 1; i < 8; ++i)
    if (samples[i] != samples[0])
      distinct = 1;
  EXPECT(distinct);

  store_destroy(latency);
  store_destroy(inner);
  fixture_rm_tree(root);
  return 0;
}

static int
test_invalid_model_rejected(void)
{
  char root[64];
  EXPECT(make_root(root, sizeof root) == 0);
  struct store* inner =
    store_fs_create(&(struct store_fs_config){ .root = root, .nthreads = 1 });
  EXPECT(inner);
  struct store* latency = store_latency_create(
    inner, &(struct damacy_latency_model){ .lognormal_sigma_ln_ns = -1.0 });
  EXPECT(latency == NULL);
  store_destroy(inner);
  fixture_rm_tree(root);
  return 0;
}

int
main(void)
{
  RUN(test_submit_map_and_stat_pay_latency_floor);
  RUN(test_lognormal_samples_vary);
  RUN(test_invalid_model_rejected);
  return 0;
}
