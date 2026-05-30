// Render job: planner output plus dispatch cursor for one sealed batch.
//
// Batch slots own caller-visible lifetime and the output tensor. A render
// job owns the work description that turns a sealed batch into one or more
// waves: read ops, chunk/sample plans, read groups, path intern table, and
// the cursor used while reserving waves.
#pragma once

#include "damacy.h"
#include "damacy_limits.h"
#include "planner/planner.h"
#include "util/path_intern.h"

#include <stdint.h>

enum render_job_state
{
  RENDER_JOB_FREE = 0,
  RENDER_JOB_READY,
};

struct render_job
{
  enum render_job_state state;
  uint16_t batch_pool_slot;
  uint64_t batch_id;

  struct read_op* read_ops;
  struct chunk_plan* chunk_plans;
  struct read_op_group* read_op_groups;
  struct sample_plan* sample_plans;
  struct path_intern paths;
  void* d_sample_plans;

  uint32_t n_read_op_groups;
  uint32_t n_sample_plans;
  uint32_t n_chunks;
  uint32_t n_chunks_to_load;
  uint32_t n_loads_issued;

  uint32_t n_chunks_dispatched;
  uint32_t n_groups_dispatched;
};

struct render_job_pool
{
  // V1 ownership: each batch slot has one paired render job with the same
  // index. Keep all direct indexing behind the helpers below.
  struct render_job jobs[DAMACY_N_BATCH_SLOTS];
};

struct store_read;

struct wave_desc
{
  uint16_t render_job_idx;
  uint16_t batch_pool_slot;
  uint32_t batch_chunk_offset;
  uint32_t n_chunks;
  uint32_t n_reads;
  uint32_t prev_n_groups_dispatched;
  uint64_t host_used_bytes;
  uint64_t io_bytes;
  uint8_t is_fill_wave;
};

struct wave_pack_limits
{
  uint64_t input_cap;
  uint64_t dev_decompressed_cap;
  uint32_t max_chunks_per_wave;
};

int
render_job_init(struct render_job* job, uint32_t samples_per_batch_cap);

void
render_job_destroy(struct render_job* job, int cuda_skip);

void
render_job_pool_destroy(struct render_job_pool* pool, int cuda_skip);

struct render_job*
render_job_pool_for_batch_slot(struct render_job_pool* pool,
                               uint16_t batch_slot_idx);

const struct render_job*
render_job_pool_for_batch_slot_const(const struct render_job_pool* pool,
                                     uint16_t batch_slot_idx);

struct render_job*
render_job_pool_get(struct render_job_pool* pool, uint16_t render_job_idx);

const struct render_job*
render_job_pool_get_const(const struct render_job_pool* pool,
                          uint16_t render_job_idx);

void
render_job_reset(struct render_job* job);

struct planner_output
render_job_planner_output(struct render_job* job, uint32_t samples_per_batch);

enum damacy_status
render_job_upload_sample_plans(struct render_job* job);

void
render_job_commit_plan(struct render_job* job,
                       uint16_t batch_pool_slot,
                       uint64_t batch_id,
                       const struct planner_output* out);

int
render_job_has_work(const struct render_job* job);

int
find_render_job_with_work(const struct render_job_pool* pool);

enum damacy_status
wave_dispatcher_reserve(struct render_job* job,
                        uint16_t render_job_idx,
                        const struct wave_pack_limits* limits,
                        struct store_read* reads,
                        void* input_dst,
                        struct wave_desc* out);

void
render_job_rollback_wave(struct render_job* job, const struct wave_desc* desc);

void
render_job_finish(struct render_job* job);
