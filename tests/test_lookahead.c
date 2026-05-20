#include "expect.h"
#include "lookahead/lookahead.h"

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

int
main(void)
{
  RUN(test_init_destroy);
  RUN(test_push_drain);
  RUN(test_full_returns_1);
  RUN(test_destroy_frees);
  RUN(test_wraparound);
  printf("all lookahead tests passed\n");
  return 0;
}
