#include "planner/coalesce.h"

#include "damacy_limits.h" // DAMACY_MAX_PATH
#include "planner/read_op_sort.h"

#include <string.h>

enum damacy_status
coalesce_chunks(struct planner_output* out,
                uint64_t read_op_max_bytes,
                uint32_t* u32_scratch,
                struct read_op* read_op_scratch)
{
  if (!out || !out->read_ops || !out->chunk_plans)
    return DAMACY_INVAL;
  out->n_chunks_to_load = 0;
  out->n_loads_issued = 0;

  uint32_t n = out->n_read_ops;
  if (n == 0)
    return DAMACY_OK;
  if (!u32_scratch || !read_op_scratch)
    return DAMACY_INVAL;

  uint32_t* perm = u32_scratch;
  uint32_t* remap = u32_scratch + n;
  uint32_t* offset_shift = u32_scratch + 2u * n;
  struct read_op* tmp = read_op_scratch;

  for (uint32_t i = 0; i < n; ++i)
    remap[i] = UINT32_MAX;

  // Partition: real (path non-empty, nbytes > 0) vs fill placeholders.
  uint32_t n_io = 0;
  for (uint32_t i = 0; i < n; ++i) {
    struct read_op* r = &out->read_ops[i];
    if (r->nbytes != 0 && r->shard_path[0] != '\0')
      perm[n_io++] = i;
  }

  read_op_perm_sort(out->read_ops, perm, n_io);

  // Greedy fuse-with-cap. A new leader starts when: path differs, or
  // there's a gap (curr_offset > leader_end), or fusing would push
  // the leader past read_op_max_bytes.
  uint32_t write = 0;
  uint32_t leader_old = UINT32_MAX;
  uint64_t leader_end = 0;
  for (uint32_t k = 0; k < n_io; ++k) {
    uint32_t e = perm[k];
    struct read_op* curr = &out->read_ops[e];
    uint64_t curr_end = (uint64_t)curr->file_offset + curr->nbytes;
    int fusable = 0;
    if (leader_old != UINT32_MAX) {
      struct read_op* leader = &out->read_ops[leader_old];
      if (strncmp(curr->shard_path, leader->shard_path, DAMACY_MAX_PATH) == 0 &&
          curr->file_offset >= leader->file_offset &&
          curr->file_offset <= leader_end) {
        uint64_t fused_end = curr_end > leader_end ? curr_end : leader_end;
        uint64_t fused_size = fused_end - leader->file_offset;
        if (fused_size <= read_op_max_bytes && fused_size <= UINT32_MAX)
          fusable = 1;
      }
    }
    if (fusable) {
      uint32_t new_idx = remap[leader_old];
      uint64_t fused_end = curr_end > leader_end ? curr_end : leader_end;
      tmp[new_idx].nbytes = (uint32_t)(fused_end - tmp[new_idx].file_offset);
      remap[e] = new_idx;
      offset_shift[e] =
        (uint32_t)(curr->file_offset - tmp[new_idx].file_offset);
      leader_end = fused_end;
    } else {
      tmp[write] = *curr;
      remap[e] = write;
      offset_shift[e] = 0;
      leader_old = e;
      leader_end = curr_end;
      write++;
    }
  }

  // Fill placeholders (path empty / nbytes == 0): keep 1:1, append at
  // the end of the output. chunk_plans referencing them still find
  // their entry via remap.
  for (uint32_t i = 0; i < n; ++i) {
    if (remap[i] != UINT32_MAX)
      continue;
    tmp[write] = out->read_ops[i];
    remap[i] = write;
    offset_shift[i] = 0;
    write++;
  }

  memcpy(out->read_ops, tmp, (size_t)write * sizeof(struct read_op));
  out->n_read_ops = write;

  for (uint32_t i = 0; i < out->n_chunk_plans; ++i) {
    struct chunk_plan* cp = &out->chunk_plans[i];
    uint32_t old = cp->read_op_idx;
    cp->read_op_idx = remap[old];
    if (cp->is_fill)
      continue;
    // Defends an invariant: fusion caps fused_size at UINT32_MAX, and
    // offset_in_read + offset_shift <= fused_size by construction.
    uint64_t sum = (uint64_t)cp->offset_in_read + (uint64_t)offset_shift[old];
    if (sum > UINT32_MAX)
      return DAMACY_INVAL;
    cp->offset_in_read = (uint32_t)sum;
  }

  // Counters for the bench's filter→fuse row.
  {
    uint32_t to_load = 0;
    for (uint32_t i = 0; i < out->n_chunk_plans; ++i)
      if (!out->chunk_plans[i].is_fill)
        to_load++;
    uint32_t loads = 0;
    for (uint32_t i = 0; i < out->n_read_ops; ++i) {
      const struct read_op* r = &out->read_ops[i];
      if (r->nbytes != 0 && r->shard_path[0] != '\0')
        loads++;
    }
    out->n_chunks_to_load = to_load;
    out->n_loads_issued = loads;
  }
  return DAMACY_OK;
}
