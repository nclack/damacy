#include "render_job.h"

#include "log/log.h"
#include "store/store.h"
#include "util/cuda_check.h"
#include "util/prelude.h"

#include <cuda.h>
#include <stdlib.h>
#include <string.h>

int
render_job_init(struct render_job* job, uint32_t samples_per_batch_cap)
{
  memset(job, 0, sizeof(*job));
  job->read_ops = (struct read_op*)calloc(DAMACY_MAX_CHUNKS_PER_BATCH,
                                          sizeof(struct read_op));
  CHECK(Error, job->read_ops);
  job->chunk_plans = (struct chunk_plan*)calloc(DAMACY_MAX_CHUNKS_PER_BATCH,
                                                sizeof(struct chunk_plan));
  CHECK(Error, job->chunk_plans);
  job->read_op_groups = (struct read_op_group*)calloc(
    DAMACY_MAX_CHUNKS_PER_BATCH, sizeof(struct read_op_group));
  CHECK(Error, job->read_op_groups);
  job->sample_plans = (struct sample_plan*)calloc(samples_per_batch_cap,
                                                  sizeof(struct sample_plan));
  CHECK(Error, job->sample_plans);
  CUdeviceptr dptr = 0;
  if (cuMemAlloc(&dptr,
                 (size_t)samples_per_batch_cap * sizeof(struct sample_plan)) !=
      CUDA_SUCCESS)
    goto Error;
  job->d_sample_plans = (void*)(uintptr_t)dptr;
  return 0;
Error:
  render_job_destroy(job, 0);
  return 1;
}

void
render_job_destroy(struct render_job* job, int cuda_skip)
{
  if (!job)
    return;
  free(job->read_ops);
  free(job->chunk_plans);
  free(job->read_op_groups);
  free(job->sample_plans);
  path_intern_free(&job->paths);
  if (!cuda_skip && job->d_sample_plans)
    cuMemFree(CUDPTR(job->d_sample_plans));
  memset(job, 0, sizeof(*job));
}

void
render_job_pool_destroy(struct render_job_pool* pool, int cuda_skip)
{
  if (!pool)
    return;
  for (int i = 0; i < 2; ++i)
    render_job_destroy(&pool->jobs[i], cuda_skip);
}

void
render_job_reset(struct render_job* job)
{
  if (!job)
    return;
  job->state = RENDER_JOB_FREE;
  job->batch_pool_slot = 0;
  job->batch_id = 0;
  job->n_read_op_groups = 0;
  job->n_sample_plans = 0;
  job->n_chunks = 0;
  job->n_chunks_to_load = 0;
  job->n_loads_issued = 0;
  job->n_chunks_dispatched = 0;
  job->n_groups_dispatched = 0;
}

struct planner_output
render_job_planner_output(struct render_job* job, uint32_t samples_per_batch)
{
  return (struct planner_output){
    .read_ops = job->read_ops,
    .read_ops_cap = DAMACY_MAX_CHUNKS_PER_BATCH,
    .chunk_plans = job->chunk_plans,
    .chunk_plans_cap = DAMACY_MAX_CHUNKS_PER_BATCH,
    .sample_plans = job->sample_plans,
    .sample_plans_cap = samples_per_batch,
    .read_op_groups = job->read_op_groups,
    .read_op_groups_cap = DAMACY_MAX_CHUNKS_PER_BATCH,
    .paths = &job->paths,
  };
}

enum damacy_status
render_job_upload_sample_plans(struct render_job* job)
{
  if (job->n_sample_plans == 0)
    return DAMACY_OK;
  return cuMemcpyHtoD(CUDPTR(job->d_sample_plans),
                      job->sample_plans,
                      (size_t)job->n_sample_plans *
                        sizeof(struct sample_plan)) == CUDA_SUCCESS
           ? DAMACY_OK
           : DAMACY_CUDA;
}

void
render_job_commit_plan(struct render_job* job,
                       uint16_t batch_pool_slot,
                       uint64_t batch_id,
                       const struct planner_output* out)
{
  job->batch_pool_slot = batch_pool_slot;
  job->batch_id = batch_id;
  job->n_chunks = out->n_chunk_plans;
  job->n_chunks_to_load = out->n_chunks_to_load;
  job->n_loads_issued = out->n_loads_issued;
  job->n_sample_plans = out->n_sample_plans;
  job->n_read_op_groups = out->n_read_op_groups;
  job->n_chunks_dispatched = 0;
  job->n_groups_dispatched = 0;
  job->state = RENDER_JOB_READY;
}

int
render_job_has_work(const struct render_job* job)
{
  return job && job->state == RENDER_JOB_READY &&
         job->n_chunks_dispatched < job->n_chunks;
}

int
find_render_job_with_work(const struct render_job_pool* pool)
{
  int best = -1;
  uint64_t best_id = UINT64_MAX;
  for (int i = 0; i < 2; ++i) {
    const struct render_job* job = &pool->jobs[i];
    if (render_job_has_work(job) && job->batch_id < best_id) {
      best = i;
      best_id = job->batch_id;
    }
  }
  return best;
}

enum damacy_status
wave_dispatcher_reserve(struct render_job* job,
                        uint16_t render_job_idx,
                        const struct wave_pack_limits* limits,
                        struct store_read* reads,
                        void* host_dst,
                        void* dev_dst,
                        struct wave_desc* out)
{
  *out =
    (struct wave_desc){ .render_job_idx = render_job_idx,
                        .batch_pool_slot = job->batch_pool_slot,
                        .batch_chunk_offset = job->n_chunks_dispatched,
                        .prev_n_groups_dispatched = job->n_groups_dispatched };
  if (!render_job_has_work(job))
    return DAMACY_OK;

  uint64_t host_cursor = 0;
  uint64_t dev_cursor = 0;
  uint32_t take = 0;
  uint32_t n_reads = 0;
  uint32_t completed_groups = job->n_groups_dispatched;

  struct read_op_group_iterator it;
  read_op_group_iterator_init(
    &it, job->read_op_groups, job->n_read_op_groups, job->n_groups_dispatched);
  struct read_op_group g;
  while (read_op_group_iterator_next(&it, &g)) {
    struct read_op* r = &job->read_ops[g.read_op_idx];
    int is_fill_group = job->chunk_plans[g.first_chunk].is_fill;
    uint64_t host_add = is_fill_group ? 0 : r->nbytes;
    if (host_cursor + host_add > limits->host_cap)
      break;
    if (take + g.n_chunks > limits->max_chunks_per_wave)
      break;
    if (dev_cursor + g.total_decompressed > limits->dev_decompressed_cap)
      break;

    uint64_t reserved_host_off = host_cursor;
    if (!is_fill_group) {
      void* dst = limits->use_gds ? (void*)((uint8_t*)dev_dst + host_cursor)
                                  : (void*)((uint8_t*)host_dst + host_cursor);
      reads[n_reads++] = (struct store_read){
        .key = r->shard_path,
        .dst = dst,
        .offset = r->file_offset,
        .len = r->nbytes,
      };
      host_cursor += host_add;
    }
    for (uint32_t i = 0; i < g.n_chunks; ++i) {
      struct chunk_plan* c = &job->chunk_plans[g.first_chunk + i];
      c->host_buf_offset = is_fill_group ? 0 : reserved_host_off;
      c->dev_decompressed_offset = dev_cursor;
      dev_cursor += c->decompressed_nbytes;
      take++;
    }
    completed_groups++;
  }

  if (take == 0) {
    const struct read_op_group* g0 =
      (job->n_groups_dispatched < job->n_read_op_groups)
        ? &job->read_op_groups[job->n_groups_dispatched]
        : NULL;
    log_error("wave: group too large for slot "
              "(group n_chunks=%u total_decompressed=%llu; "
              "slot_cap=%llu dev_cap=%llu)",
              g0 ? g0->n_chunks : 0u,
              g0 ? (unsigned long long)g0->total_decompressed : 0ull,
              (unsigned long long)limits->host_cap,
              (unsigned long long)limits->dev_decompressed_cap);
    return DAMACY_BUDGET;
  }

  out->n_chunks = take;
  out->n_reads = n_reads;
  out->host_used_bytes = host_cursor;
  out->io_bytes = host_cursor;
  out->is_fill_wave = (uint8_t)(n_reads == 0);
  job->n_chunks_dispatched += take;
  job->n_groups_dispatched = completed_groups;
  return DAMACY_OK;
}

void
render_job_rollback_wave(struct render_job* job, const struct wave_desc* desc)
{
  if (!job || !desc)
    return;
  job->n_chunks_dispatched -= desc->n_chunks;
  job->n_groups_dispatched = desc->prev_n_groups_dispatched;
}

void
render_job_finish(struct render_job* job)
{
  render_job_reset(job);
}
