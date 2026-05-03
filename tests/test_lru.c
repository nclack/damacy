#include "expect.h"
#include "util/hash.h"
#include "util/lru.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Test value: a heap-allocated string. eq compares string equality.
struct sval
{
  char* s;
};

static struct sval*
sval_new(const char* s)
{
  struct sval* v = (struct sval*)malloc(sizeof(*v));
  v->s = strdup(s);
  return v;
}

static int
sval_eq(const void* value, const void* probe_key, void* user)
{
  (void)user;
  const struct sval* v = (const struct sval*)value;
  const char* k = (const char*)probe_key;
  return strcmp(v->s, k) == 0;
}

static uint64_t destroy_count = 0;
static void
sval_destroy(void* value, void* user)
{
  (void)user;
  ++destroy_count;
  struct sval* v = (struct sval*)value;
  free(v->s);
  free(v);
}

static const struct lru_ops SVAL_OPS = {
  .eq = sval_eq,
  .destroy = sval_destroy,
  .user = NULL,
};

static int
test_basic_put_get(void)
{
  destroy_count = 0;
  struct lru* l = lru_create(8, 16, &SVAL_OPS);
  EXPECT(l);

  struct lru_entry* e1 =
    lru_put(l, hash_fnv1a_str("alpha"), "alpha", sval_new("alpha"));
  EXPECT(e1);
  struct lru_entry* e2 =
    lru_put(l, hash_fnv1a_str("beta"), "beta", sval_new("beta"));
  EXPECT(e2);

  struct lru_entry* g = lru_get(l, hash_fnv1a_str("alpha"), "alpha");
  EXPECT(g == e1);
  EXPECT(strcmp(((struct sval*)lru_entry_value(g))->s, "alpha") == 0);

  EXPECT(lru_get(l, hash_fnv1a_str("missing"), "missing") == NULL);

  struct lru_stats st;
  lru_stats_get(l, &st);
  EXPECT(st.counters.hits == 1);
  EXPECT(st.counters.misses == 1);
  EXPECT(st.counters.insertions == 2);
  EXPECT(st.size == 2);
  EXPECT(st.counters.evictions == 0);

  lru_destroy(l);
  EXPECT(destroy_count == 2);
  return 0;
}

static int
test_replace_same_key(void)
{
  destroy_count = 0;
  struct lru* l = lru_create(8, 16, &SVAL_OPS);
  EXPECT(l);

  uint64_t h = hash_fnv1a_str("alpha");
  lru_put(l, h, "alpha", sval_new("alpha"));
  // Re-put same key with a fresh value: old must be destroyed, count == 1.
  lru_put(l, h, "alpha", sval_new("alpha"));
  EXPECT(destroy_count == 1);

  struct lru_stats st;
  lru_stats_get(l, &st);
  EXPECT(st.size == 1);
  EXPECT(st.counters.replacements == 1);
  EXPECT(st.counters.insertions == 1); // only the first put counted as insert

  lru_destroy(l);
  EXPECT(destroy_count == 2);
  return 0;
}

static int
test_eviction_oldest_first(void)
{
  destroy_count = 0;
  struct lru* l = lru_create(4, 16, &SVAL_OPS);
  EXPECT(l);

  const char* keys[] = { "a", "b", "c", "d" };
  for (uint32_t i = 0; i < 4; ++i)
    lru_put(l, hash_fnv1a_str(keys[i]), keys[i], sval_new(keys[i]));

  // Touch "a" so it's MRU; "b" becomes LRU tail.
  EXPECT(lru_get(l, hash_fnv1a_str("a"), "a") != NULL);

  // Insert "e": one eviction; victim should be "b".
  lru_put(l, hash_fnv1a_str("e"), "e", sval_new("e"));

  EXPECT(lru_get(l, hash_fnv1a_str("a"), "a") != NULL);
  EXPECT(lru_get(l, hash_fnv1a_str("b"), "b") == NULL); // evicted
  EXPECT(lru_get(l, hash_fnv1a_str("c"), "c") != NULL);
  EXPECT(lru_get(l, hash_fnv1a_str("d"), "d") != NULL);
  EXPECT(lru_get(l, hash_fnv1a_str("e"), "e") != NULL);

  struct lru_stats st;
  lru_stats_get(l, &st);
  EXPECT(st.size == 4);
  EXPECT(st.counters.evictions == 1);

  lru_destroy(l);
  return 0;
}

static int
test_pinning_skips_eviction(void)
{
  destroy_count = 0;
  struct lru* l = lru_create(3, 16, &SVAL_OPS);
  EXPECT(l);

  struct lru_entry* ea = lru_put(l, hash_fnv1a_str("a"), "a", sval_new("a"));
  lru_put(l, hash_fnv1a_str("b"), "b", sval_new("b"));
  lru_put(l, hash_fnv1a_str("c"), "c", sval_new("c"));

  // Pin "a" — it's the LRU tail.
  lru_entry_acquire(ea);

  // Insert "d": eviction must skip pinned "a", evict "b" instead.
  lru_put(l, hash_fnv1a_str("d"), "d", sval_new("d"));

  EXPECT(lru_get(l, hash_fnv1a_str("a"), "a") == ea);
  EXPECT(lru_get(l, hash_fnv1a_str("b"), "b") == NULL); // evicted
  EXPECT(lru_get(l, hash_fnv1a_str("c"), "c") != NULL);
  EXPECT(lru_get(l, hash_fnv1a_str("d"), "d") != NULL);

  // Release pin; subsequent eviction may now pick "a".
  lru_entry_release(ea);

  lru_destroy(l);
  return 0;
}

static int
test_pinning_all_pinned_returns_null(void)
{
  destroy_count = 0;
  struct lru* l = lru_create(2, 16, &SVAL_OPS);
  EXPECT(l);

  struct lru_entry* ea = lru_put(l, hash_fnv1a_str("a"), "a", sval_new("a"));
  struct lru_entry* eb = lru_put(l, hash_fnv1a_str("b"), "b", sval_new("b"));
  lru_entry_acquire(ea);
  lru_entry_acquire(eb);

  // Insert "c": cannot evict; both pinned. lru_put must return NULL and
  // destroy our value.
  uint64_t before = destroy_count;
  struct lru_entry* ec = lru_put(l, hash_fnv1a_str("c"), "c", sval_new("c"));
  EXPECT(ec == NULL);
  EXPECT(destroy_count == before + 1);

  struct lru_stats st;
  lru_stats_get(l, &st);
  EXPECT(st.counters.put_failures == 1);
  EXPECT(st.size == 2);

  lru_entry_release(ea);
  lru_entry_release(eb);
  lru_destroy(l);
  return 0;
}

static int
test_collision_chain(void)
{
  destroy_count = 0;
  // Force collisions by giving everything the same hash.
  struct lru* l = lru_create(8, 16, &SVAL_OPS);
  EXPECT(l);

  const char* keys[] = { "a", "b", "c", "d", "e" };
  for (uint32_t i = 0; i < 5; ++i)
    lru_put(l, 0xdeadbeefdeadbeefull, keys[i], sval_new(keys[i]));

  // All 5 should be retrievable despite identical hashes.
  for (uint32_t i = 0; i < 5; ++i) {
    struct lru_entry* g = lru_get(l, 0xdeadbeefdeadbeefull, keys[i]);
    EXPECT(g);
    EXPECT(strcmp(((struct sval*)lru_entry_value(g))->s, keys[i]) == 0);
  }

  struct lru_stats st;
  lru_stats_get(l, &st);
  EXPECT(st.size == 5);
  EXPECT(st.max_probe_observed >= 4); // last insertion probed past 4 occupants

  lru_destroy(l);
  return 0;
}

static int
test_backshift_keeps_lookup_terminating(void)
{
  destroy_count = 0;
  struct lru* l = lru_create(4, 16, &SVAL_OPS);
  EXPECT(l);

  // Insert 4 entries with the same hash (forces a chain), then evict
  // the LRU tail and verify all remaining lookups still succeed (i.e.,
  // backshift correctly removed the entry without breaking the chain).
  const char* keys[] = { "a", "b", "c", "d" };
  for (uint32_t i = 0; i < 4; ++i)
    lru_put(l, 0xcafebabeull, keys[i], sval_new(keys[i]));

  // Insert a 5th colliding entry: forces eviction of "a" (LRU tail).
  lru_put(l, 0xcafebabeull, "e", sval_new("e"));

  EXPECT(lru_get(l, 0xcafebabeull, "a") == NULL);
  for (uint32_t i = 1; i < 4; ++i)
    EXPECT(lru_get(l, 0xcafebabeull, keys[i]) != NULL);
  EXPECT(lru_get(l, 0xcafebabeull, "e") != NULL);

  lru_destroy(l);
  return 0;
}

// Robin-hood should keep probe distances tight under realistic
// (well-distributed) hash inputs. Insert N entries with FNV-1a hashes
// of distinct strings; with load factor 50% (idx_size = 2 * capacity)
// the observed max probe should stay well under log2(capacity) + a
// small safety constant.
static int
test_robin_hood_redistributes(void)
{
  destroy_count = 0;
  struct lru* l = lru_create(64, 16, &SVAL_OPS);
  EXPECT(l);

  for (int i = 0; i < 64; ++i) {
    char buf[16];
    snprintf(buf, sizeof buf, "key-%d", i);
    EXPECT(lru_put(l, hash_fnv1a_str(buf), buf, sval_new(buf)) != NULL);
  }

  struct lru_stats st;
  lru_stats_get(l, &st);
  EXPECT(st.size == 64);
  // 64 entries in idx_size 128 (50% load) + good hash distribution:
  // robin-hood worst-case probe should stay small. Empirical bound for
  // FNV-1a on these strings is well under 8; assert < 12 for headroom.
  EXPECT(st.max_probe_observed < 12);

  // Every insert is still retrievable.
  for (int i = 0; i < 64; ++i) {
    char buf[16];
    snprintf(buf, sizeof buf, "key-%d", i);
    EXPECT(lru_get(l, hash_fnv1a_str(buf), buf) != NULL);
  }

  lru_destroy(l);
  return 0;
}

int
main(void)
{
  RUN(test_basic_put_get);
  RUN(test_replace_same_key);
  RUN(test_eviction_oldest_first);
  RUN(test_pinning_skips_eviction);
  RUN(test_pinning_all_pinned_returns_null);
  RUN(test_collision_chain);
  RUN(test_backshift_keeps_lookup_terminating);
  RUN(test_robin_hood_redistributes);
  log_info("all tests passed");
  return 0;
}
