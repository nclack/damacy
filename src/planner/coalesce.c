#include "planner/coalesce.h"

#include "planner/read_op_sort.h"

enum damacy_status
coalesce_chunks(struct planner_output* out,
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

  uint32_t* perm = u32_scratch;
  uint32_t* remap = u32_scratch + n;
  uint32_t* offset_shift = u32_scratch + 2u * n;
  struct read_op* tmp = read_op_scratch;

  for (uint32_t i = 0; i < n; ++i)
    remap[i] = UINT32_MAX;

  // Partition: real (path interned, nbytes > 0) vs fill placeholders.
  uint32_t n_io = 0;
  for (uint32_t i = 0; i < n; ++i) {
    struct read_op* r = &out->read_ops[i];
    if (r->nbytes != 0 && r->shard_path)
      perm[n_io++] = i;
  }

  read_op_perm_sort(out->read_ops, perm, n_io);

  // leader_chunks = post-coalesce group size; capped so each group
  // fits one wave's chunk intake (planner is 1 read_op per chunk).
  uint32_t write = 0;
  uint32_t leader_old = UINT32_MAX;
  uint64_t leader_end = 0;
  uint32_t leader_chunks = 0;
  for (uint32_t k = 0; k < n_io; ++k) {
    uint32_t e = perm[k];
    struct read_op* curr = &out->read_ops[e];
    uint64_t curr_end = (uint64_t)curr->file_offset + curr->nbytes;
    int fusable = 0;
    if (leader_old != UINT32_MAX) {
      struct read_op* leader = &out->read_ops[leader_old];
      if (curr->shard_path == leader->shard_path &&
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
      uint32_t new_idx = remap[leader_old];
      uint64_t fused_end = curr_end > leader_end ? curr_end : leader_end;
      tmp[new_idx].nbytes = (uint32_t)(fused_end - tmp[new_idx].file_offset);
      remap[e] = new_idx;
      offset_shift[e] =
        (uint32_t)(curr->file_offset - tmp[new_idx].file_offset);
      leader_end = fused_end;
      leader_chunks++;
    } else {
      tmp[write] = *curr;
      remap[e] = write;
      offset_shift[e] = 0;
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
  uint32_t* pos = u32_scratch + 3u * n;
  {
    uint32_t* starts = perm; // perm is dead after the fuse loop
    uint32_t n_runs = 0;
    for (uint32_t k = 0; k < n_real; ++k)
      if (k == 0 || tmp[k].shard_path != tmp[k - 1].shard_path)
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
  // the end of the output. chunk_plans referencing them still find
  // their entry via remap.
  for (uint32_t i = 0; i < n; ++i) {
    if (remap[i] != UINT32_MAX)
      continue;
    tmp[write] = out->read_ops[i];
    remap[i] = write;
    offset_shift[i] = 0;
    pos[write] = write;
    write++;
  }

  for (uint32_t k = 0; k < write; ++k)
    out->read_ops[pos[k]] = tmp[k];
  out->n_read_ops = write;

  for (uint32_t i = 0; i < out->n_chunk_plans; ++i) {
    struct chunk_plan* cp = &out->chunk_plans[i];
    uint32_t old = cp->read_op_idx;
    cp->read_op_idx = pos[remap[old]];
    if (cp->is_fill)
      continue;
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
      if (r->nbytes != 0 && r->shard_path)
        loads++;
    }
    out->n_chunks_to_load = to_load;
    out->n_loads_issued = loads;
  }
  return DAMACY_OK;
}
