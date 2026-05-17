#include "planner/read_op_sort.h"

#include "damacy_limits.h"   // DAMACY_MAX_PATH
#include "planner/planner.h" // struct read_op

#include <string.h>

#define READ_OP_SORT_INSERTION_CUTOFF 16u

static int
perm_lt(const struct read_op* ops, uint32_t a, uint32_t b)
{
  int cmp = strncmp(ops[a].shard_path, ops[b].shard_path, DAMACY_MAX_PATH);
  if (cmp != 0)
    return cmp < 0;
  if (ops[a].file_offset != ops[b].file_offset)
    return ops[a].file_offset < ops[b].file_offset;
  return a < b;
}

static void
perm_insertion_sort(const struct read_op* ops,
                    uint32_t* perm,
                    uint32_t lo,
                    uint32_t hi)
{
  for (uint32_t i = lo + 1; i <= hi; ++i) {
    uint32_t key = perm[i];
    uint32_t j = i;
    while (j > lo && !perm_lt(ops, perm[j - 1], key)) {
      perm[j] = perm[j - 1];
      j--;
    }
    perm[j] = key;
  }
}

// Recurses on the smaller partition, iterates on the larger — keeps
// stack depth O(log n) on adversarial inputs.
static void
perm_quicksort(const struct read_op* ops,
               uint32_t* perm,
               uint32_t lo,
               uint32_t hi)
{
  while (hi > lo) {
    if (hi - lo < READ_OP_SORT_INSERTION_CUTOFF) {
      perm_insertion_sort(ops, perm, lo, hi);
      return;
    }
    uint32_t mid = lo + (hi - lo) / 2u;
    if (perm_lt(ops, perm[mid], perm[lo])) {
      uint32_t t = perm[lo];
      perm[lo] = perm[mid];
      perm[mid] = t;
    }
    if (perm_lt(ops, perm[hi], perm[lo])) {
      uint32_t t = perm[lo];
      perm[lo] = perm[hi];
      perm[hi] = t;
    }
    if (perm_lt(ops, perm[hi], perm[mid])) {
      uint32_t t = perm[mid];
      perm[mid] = perm[hi];
      perm[hi] = t;
    }
    uint32_t pivot_idx = hi - 1u;
    {
      uint32_t t = perm[mid];
      perm[mid] = perm[pivot_idx];
      perm[pivot_idx] = t;
    }
    uint32_t pivot = perm[pivot_idx];
    uint32_t i = lo;
    uint32_t j = pivot_idx;
    for (;;) {
      while (perm_lt(ops, perm[++i], pivot))
        ;
      while (perm_lt(ops, pivot, perm[--j]))
        ;
      if (i >= j)
        break;
      uint32_t t = perm[i];
      perm[i] = perm[j];
      perm[j] = t;
    }
    {
      uint32_t t = perm[i];
      perm[i] = perm[pivot_idx];
      perm[pivot_idx] = t;
    }
    if (i > 0u && (i - 1u) - lo < hi - (i + 1u)) {
      if (i > lo + 1u)
        perm_quicksort(ops, perm, lo, i - 1u);
      lo = i + 1u;
    } else {
      if (i + 1u < hi)
        perm_quicksort(ops, perm, i + 1u, hi);
      if (i == 0u)
        return;
      hi = i - 1u;
    }
  }
}

void
read_op_perm_sort(const struct read_op* ops, uint32_t* perm, uint32_t n)
{
  if (n > 1u)
    perm_quicksort(ops, perm, 0u, n - 1u);
}
