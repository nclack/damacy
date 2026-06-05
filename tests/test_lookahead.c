#include "expect.h"
#include "lookahead/lookahead.h"
#include "platform/platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct damacy_sample
mk_sample(const char* uri, int64_t lo, int64_t hi)
{
  struct damacy_sample s = { 0 };
  s.uri = uri;
  s.aabb.rank = 1;
  s.aabb.dims[0].beg = lo;
  s.aabb.dims[0].end = hi;
  return s;
}

static int
test_init_destroy(void)
{
  struct damacy_lookahead la = { 0 };
  EXPECT(lookahead_init(&la, 4) == 0);
  EXPECT(la.cap == 4);
  EXPECT(la.size == 0);
  lookahead_destroy(&la);
  EXPECT(la.slots == NULL);
  lookahead_destroy(&la);
  return 0;
}

static int
test_push_drain(void)
{
  struct damacy_lookahead la = { 0 };
  EXPECT(lookahead_init(&la, 4) == 0);

  struct damacy_sample s0 = mk_sample("zero", 0, 4);
  struct damacy_sample s1 = mk_sample("one", 4, 8);
  struct damacy_sample s2 = mk_sample("two", 8, 12);

  EXPECT(lookahead_push(&la, &s0) == 0);
  EXPECT(lookahead_push(&la, &s1) == 0);
  EXPECT(lookahead_push(&la, &s2) == 0);
  EXPECT(la.size == 3);

  struct damacy_sample_slot out[3] = { 0 };
  lookahead_drain(&la, out, 3);
  EXPECT(la.size == 0);
  EXPECT(strcmp(out[0].uri, "zero") == 0);
  EXPECT(strcmp(out[1].uri, "one") == 0);
  EXPECT(strcmp(out[2].uri, "two") == 0);
  EXPECT(out[0].aabb.dims[0].end == 4);
  EXPECT(out[2].aabb.dims[0].beg == 8);

  for (uint32_t i = 0; i < la.cap; ++i)
    EXPECT(la.slots[i].uri == NULL);

  for (int i = 0; i < 3; ++i)
    sample_slot_clear(&out[i]);

  lookahead_destroy(&la);
  return 0;
}

static int
test_full_returns_1(void)
{
  struct damacy_lookahead la = { 0 };
  EXPECT(lookahead_init(&la, 2) == 0);
  struct damacy_sample s = mk_sample("a", 0, 1);
  EXPECT(lookahead_push(&la, &s) == 0);
  EXPECT(lookahead_push(&la, &s) == 0);
  EXPECT(la.size == 2);
  EXPECT(lookahead_push(&la, &s) == 1);
  EXPECT(la.size == 2);
  lookahead_destroy(&la);
  return 0;
}

static int
test_destroy_frees(void)
{
  struct damacy_lookahead la = { 0 };
  EXPECT(lookahead_init(&la, 4) == 0);
  struct damacy_sample s = mk_sample("leftover", 0, 1);
  EXPECT(lookahead_push(&la, &s) == 0);
  EXPECT(lookahead_push(&la, &s) == 0);
  lookahead_destroy(&la);
  EXPECT(la.slots == NULL);
  return 0;
}

static int
test_wraparound(void)
{
  struct damacy_lookahead la = { 0 };
  EXPECT(lookahead_init(&la, 3) == 0);
  struct damacy_sample s_a = mk_sample("a", 0, 1);
  struct damacy_sample s_b = mk_sample("b", 0, 1);
  struct damacy_sample s_c = mk_sample("c", 0, 1);
  EXPECT(lookahead_push(&la, &s_a) == 0);
  EXPECT(lookahead_push(&la, &s_b) == 0);

  struct damacy_sample_slot first[2] = { 0 };
  lookahead_drain(&la, first, 2);
  for (int i = 0; i < 2; ++i)
    sample_slot_clear(&first[i]);

  EXPECT(lookahead_push(&la, &s_c) == 0);
  struct damacy_sample_slot second[1] = { 0 };
  lookahead_drain(&la, second, 1);
  EXPECT(strcmp(second[0].uri, "c") == 0);
  sample_slot_clear(&second[0]);

  lookahead_destroy(&la);
  return 0;
}

static int
test_push_with_sample_seq_round_trip(void)
{
  struct damacy_lookahead la = { 0 };
  EXPECT(lookahead_init(&la, 4) == 0);
  struct damacy_sample s = mk_sample("k", 0, 1);
  EXPECT(lookahead_push_with_sample_seq(&la, &s, 42) == 0);

  struct damacy_sample_slot out = { 0 };
  EXPECT(lookahead_pop_blocking(&la, &out) == 1);
  EXPECT(strcmp(out.uri, "k") == 0);
  EXPECT(out.sample_seq == 42);
  sample_slot_clear(&out);

  lookahead_destroy(&la);
  return 0;
}

static int
test_pop_blocking_returns_existing_after_stop(void)
{
  struct damacy_lookahead la = { 0 };
  EXPECT(lookahead_init(&la, 4) == 0);
  struct damacy_sample s = mk_sample("k", 0, 1);
  EXPECT(lookahead_push_with_sample_seq(&la, &s, 7) == 0);

  lookahead_signal_stop(&la);

  struct damacy_sample_slot out = { 0 };
  EXPECT(lookahead_pop_blocking(&la, &out) == 1);
  EXPECT(out.sample_seq == 7);
  sample_slot_clear(&out);

  EXPECT(lookahead_pop_blocking(&la, &out) == 0);

  lookahead_destroy(&la);
  return 0;
}

struct wake_ctx
{
  struct damacy_lookahead* la;
};

static void
push_after_delay(void* arg)
{
  struct wake_ctx* ctx = (struct wake_ctx*)arg;
  platform_sleep_ns(10 * 1000 * 1000);
  struct damacy_sample s = mk_sample("late", 0, 1);
  lookahead_push_with_sample_seq(ctx->la, &s, 99);
}

static int
test_pop_blocking_wakes_on_push(void)
{
  struct damacy_lookahead la = { 0 };
  EXPECT(lookahead_init(&la, 4) == 0);

  struct wake_ctx ctx = { .la = &la };
  struct platform_thread* t = platform_thread_start(push_after_delay, &ctx);
  EXPECT(t);

  struct damacy_sample_slot out = { 0 };
  EXPECT(lookahead_pop_blocking(&la, &out) == 1);
  EXPECT(strcmp(out.uri, "late") == 0);
  EXPECT(out.sample_seq == 99);
  sample_slot_clear(&out);

  platform_thread_join(t);
  lookahead_destroy(&la);
  return 0;
}

static void
pop_in_thread(void* arg)
{
  struct damacy_lookahead* la = (struct damacy_lookahead*)arg;
  struct damacy_sample_slot out = { 0 };
  (void)lookahead_pop_blocking(la, &out);
  sample_slot_clear(&out);
}

static int
test_signal_stop_unblocks_empty_pop(void)
{
  struct damacy_lookahead la = { 0 };
  EXPECT(lookahead_init(&la, 4) == 0);

  struct platform_thread* t = platform_thread_start(pop_in_thread, &la);
  EXPECT(t);
  platform_sleep_ns(5 * 1000 * 1000);
  lookahead_signal_stop(&la);
  platform_thread_join(t);

  lookahead_destroy(&la);
  return 0;
}

static int
test_pop_blocking_timeout_returns_on_timeout(void)
{
  struct damacy_lookahead la = { 0 };
  EXPECT(lookahead_init(&la, 4) == 0);

  struct damacy_sample_slot out = { 0 };
  EXPECT(lookahead_pop_blocking_timeout(&la, &out, 5) == 0);

  lookahead_destroy(&la);
  return 0;
}

static int
test_pop_blocking_timeout_pops_on_push(void)
{
  struct damacy_lookahead la = { 0 };
  EXPECT(lookahead_init(&la, 4) == 0);

  struct wake_ctx ctx = { .la = &la };
  struct platform_thread* t = platform_thread_start(push_after_delay, &ctx);
  EXPECT(t);

  struct damacy_sample_slot out = { 0 };
  EXPECT(lookahead_pop_blocking_timeout(&la, &out, 1000) == 1);
  EXPECT(strcmp(out.uri, "late") == 0);
  EXPECT(out.sample_seq == 99);
  sample_slot_clear(&out);

  platform_thread_join(t);
  lookahead_destroy(&la);
  return 0;
}

static int
test_wait_nonempty_timeout_returns_on_timeout(void)
{
  struct damacy_lookahead la = { 0 };
  EXPECT(lookahead_init(&la, 4) == 0);

  EXPECT(lookahead_wait_nonempty_timeout(&la, 5) == 0);

  lookahead_destroy(&la);
  return 0;
}

static int
test_wait_nonempty_wakes_without_consuming(void)
{
  struct damacy_lookahead la = { 0 };
  EXPECT(lookahead_init(&la, 4) == 0);

  struct wake_ctx ctx = { .la = &la };
  struct platform_thread* t = platform_thread_start(push_after_delay, &ctx);
  EXPECT(t);

  EXPECT(lookahead_wait_nonempty_timeout(&la, -1) == 1);

  struct damacy_sample_slot out = { 0 };
  EXPECT(lookahead_try_pop(&la, &out) == 1);
  EXPECT(strcmp(out.uri, "late") == 0);
  EXPECT(out.sample_seq == 99);
  sample_slot_clear(&out);

  platform_thread_join(t);
  lookahead_destroy(&la);
  return 0;
}

int
main(void)
{
  RUN(test_init_destroy);
  RUN(test_push_drain);
  RUN(test_full_returns_1);
  RUN(test_destroy_frees);
  RUN(test_wraparound);
  RUN(test_push_with_sample_seq_round_trip);
  RUN(test_pop_blocking_returns_existing_after_stop);
  RUN(test_pop_blocking_wakes_on_push);
  RUN(test_signal_stop_unblocks_empty_pop);
  RUN(test_pop_blocking_timeout_returns_on_timeout);
  RUN(test_pop_blocking_timeout_pops_on_push);
  RUN(test_wait_nonempty_timeout_returns_on_timeout);
  RUN(test_wait_nonempty_wakes_without_consuming);
  printf("all lookahead tests passed\n");
  return 0;
}
