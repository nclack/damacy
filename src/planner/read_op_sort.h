// Sort a permutation array of read_op indices by (shard_path,
// file_offset, original index). Used by the coalesce step.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct read_op;

  void read_op_perm_sort(const struct read_op* ops, uint32_t* perm, uint32_t n);

#ifdef __cplusplus
}
#endif
