#include "executor/coalesce.h"

#include "executor/read_op_sort.h"

#include <string.h>

static int
same_path(const char* a, const char* b)
{
  return a == b || !strcmp(a, b);
}

enum damacy_status
coalesce_reads(struct read_op* reads,
               uint32_t* n_reads,
               uint64_t read_op_max_bytes,
               uint32_t max_chunks_per_wave,
               uint32_t* read_index,
               uint32_t* offset_in_read,
               uint32_t* u32_scratch,
               struct read_op* read_op_scratch)
{
  if (!reads || !n_reads)
    return DAMACY_INVAL;
  uint32_t n = *n_reads;
  if (n == 0)
    return DAMACY_OK;
  if (!read_index || !offset_in_read || !u32_scratch || !read_op_scratch)
    return DAMACY_INVAL;

  uint32_t* perm = u32_scratch;
  struct read_op* tmp = read_op_scratch;

  for (uint32_t i = 0; i < n; ++i)
    read_index[i] = UINT32_MAX;

  // Partition: real (path present, nbytes > 0) vs fill placeholders.
  uint32_t n_io = 0;
  for (uint32_t i = 0; i < n; ++i) {
    struct read_op* r = &reads[i];
    if (r->nbytes != 0 && r->shard_path)
      perm[n_io++] = i;
  }

  read_op_perm_sort(reads, perm, n_io);

  // leader_chunks = post-coalesce group size; capped so each group
  // fits one wave's chunk intake (planner is 1 read_op per chunk).
  uint32_t write = 0;
  uint32_t leader_old = UINT32_MAX;
  uint64_t leader_end = 0;
  uint32_t leader_chunks = 0;
  for (uint32_t k = 0; k < n_io; ++k) {
    uint32_t e = perm[k];
    struct read_op* curr = &reads[e];
    uint64_t curr_end = (uint64_t)curr->file_offset + curr->nbytes;
    int fusable = 0;
    if (leader_old != UINT32_MAX) {
      struct read_op* leader = &reads[leader_old];
      if (same_path(curr->shard_path, leader->shard_path) &&
          curr->file_offset >= leader->file_offset &&
          curr->file_offset <= leader_end &&
          leader_chunks < max_chunks_per_wave) {
        uint64_t fused_end = curr_end > leader_end ? curr_end : leader_end;
        uint64_t fused_size = fused_end - leader->file_offset;
        if (fused_size <= read_op_max_bytes && fused_size <= UINT32_MAX)
          fusable = 1;
      }
    }
    if (fusable) {
      uint32_t new_idx = read_index[leader_old];
      uint64_t fused_end = curr_end > leader_end ? curr_end : leader_end;
      tmp[new_idx].nbytes = (uint32_t)(fused_end - tmp[new_idx].file_offset);
      read_index[e] = new_idx;
      offset_in_read[e] =
        (uint32_t)(curr->file_offset - tmp[new_idx].file_offset);
      leader_end = fused_end;
      leader_chunks++;
    } else {
      tmp[write] = *curr;
      read_index[e] = write;
      offset_in_read[e] = 0;
      leader_old = e;
      leader_end = curr_end;
      leader_chunks = 1;
      write++;
    }
  }

  uint32_t n_real = write;

  // Simultaneous reads into one file serialize on network filesystems
  // (~2x slower bulk reads), so spread the emitted order across shards.
  // pos[k] = output slot of tmp[k]; fills below get identity slots.
  uint32_t* pos = u32_scratch + n;
  {
    uint32_t* starts = perm; // perm is dead after the fuse loop
    uint32_t n_runs = 0;
    for (uint32_t k = 0; k < n_real; ++k)
      if (k == 0 || !same_path(tmp[k].shard_path, tmp[k - 1].shard_path))
        starts[n_runs++] = k;
    uint32_t outk = 0;
    for (uint32_t round = 0; outk < n_real; ++round) {
      for (uint32_t ri = 0; ri < n_runs; ++ri) {
        uint32_t k = starts[ri] + round;
        uint32_t end = ri + 1 < n_runs ? starts[ri + 1] : n_real;
        if (k < end)
          pos[k] = outk++;
      }
    }
  }

  // Fill placeholders (path empty / nbytes == 0): keep 1:1, append at
  // the end of the output.
  for (uint32_t i = 0; i < n; ++i) {
    if (read_index[i] != UINT32_MAX)
      continue;
    tmp[write] = reads[i];
    read_index[i] = write;
    offset_in_read[i] = 0;
    pos[write] = write;
    write++;
  }

  for (uint32_t k = 0; k < write; ++k)
    reads[pos[k]] = tmp[k];
  for (uint32_t i = 0; i < n; ++i)
    read_index[i] = pos[read_index[i]];
  *n_reads = write;
  return DAMACY_OK;
}

enum damacy_status
coalesce_chunks(struct dispatch_output* out,
                uint64_t read_op_max_bytes,
                uint32_t max_chunks_per_wave,
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

  uint32_t* read_index = u32_scratch + 2u * n;
  uint32_t* offset_in_read = u32_scratch + 3u * n;
  enum damacy_status status = coalesce_reads(out->read_ops,
                                             &out->n_read_ops,
                                             read_op_max_bytes,
                                             max_chunks_per_wave,
                                             read_index,
                                             offset_in_read,
                                             u32_scratch,
                                             read_op_scratch);
  if (status != DAMACY_OK)
    return status;

  for (uint32_t i = 0; i < out->n_chunk_plans; ++i) {
    struct chunk_plan* cp = &out->chunk_plans[i];
    uint32_t old = cp->read_op_idx;
    cp->read_op_idx = read_index[old];
    if (cp->is_fill)
      continue;
    uint64_t sum = (uint64_t)cp->offset_in_read + (uint64_t)offset_in_read[old];
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
      if (r->nbytes != 0 && r->shard_path)
        loads++;
    }
    out->n_chunks_to_load = to_load;
    out->n_loads_issued = loads;
  }
  return DAMACY_OK;
}
