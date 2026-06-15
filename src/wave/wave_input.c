#include "wave_input.h"

#include "log/log.h"
#include "wave_pool.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

// DAMACY_TRACE_WAVES=<file>: append one line per reserved wave:
// "<batch_id> <render_job_idx> <n_reads> <distinct_shards> <n_chunks>
//  <input_bytes> <stop_reason>". Probe for #154 — distinct_shards is the
// io width the coalescer's round-robin achieves for this wave.
static FILE* g_wave_trace;
static pthread_mutex_t g_wave_trace_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t g_wave_trace_once = PTHREAD_ONCE_INIT;

static void
wave_trace_open(void)
{
  const char* path = getenv("DAMACY_TRACE_WAVES");
  if (path && path[0])
    g_wave_trace = fopen(path, "a");
}

// Unique shard files among a wave's reads. The reads carry interned
// shard_path pointers (coalesce interns + round-robins them), so pointer
// identity is shard identity. O(n^2), but n_reads is small — and smallest
// exactly in the starved case this probe is chasing.
static uint32_t
wave_distinct_shards(const struct store_read* reads, uint32_t n)
{
  uint32_t distinct = 0;
  for (uint32_t i = 0; i < n; ++i) {
    int seen = 0;
    for (uint32_t j = 0; j < i; ++j)
      if (reads[j].key == reads[i].key) {
        seen = 1;
        break;
      }
    distinct += !seen;
  }
  return distinct;
}

static void
wave_account(struct wave_pool* wp,
             const struct input_slot* slot,
             uint64_t batch_id,
             const struct wave_desc* desc)
{
  uint32_t distinct = wave_distinct_shards(slot->store_reads, desc->n_reads);
  wp->stats->wave_reads_sum += desc->n_reads;
  wp->stats->wave_distinct_shards_sum += distinct;
  switch (desc->stop_reason) {
    case WAVE_STOP_HOST: wp->stats->wave_stop_host++; break;
    case WAVE_STOP_CHUNKS: wp->stats->wave_stop_chunks++; break;
    case WAVE_STOP_DEV: wp->stats->wave_stop_dev++; break;
    default: wp->stats->wave_stop_drained++; break;
  }

  pthread_once(&g_wave_trace_once, wave_trace_open);
  if (g_wave_trace) {
    pthread_mutex_lock(&g_wave_trace_lock);
    fprintf(g_wave_trace,
            "%llu %u %u %u %u %llu %u\n",
            (unsigned long long)batch_id,
            (unsigned)desc->render_job_idx,
            (unsigned)desc->n_reads,
            (unsigned)distinct,
            (unsigned)desc->n_chunks,
            (unsigned long long)desc->input_used_bytes,
            (unsigned)desc->stop_reason);
    pthread_mutex_unlock(&g_wave_trace_lock);
  }
}

static void
mark_changed(int* changed)
{
  if (changed)
    *changed = 1;
}

static struct wave_desc
input_slot_wave_desc(const struct input_slot* slot)
{
  return (struct wave_desc){ .render_job_idx = slot->render_job_idx,
                             .batch_pool_slot = slot->batch_pool_slot,
                             .batch_chunk_offset = slot->batch_chunk_offset,
                             .n_chunks = slot->n_chunks,
                             .prev_n_groups_dispatched =
                               slot->prev_n_groups_dispatched,
                             .input_used_bytes = slot->used_bytes,
                             .io_bytes = slot->io_bytes,
                             .is_fill_wave = slot->is_fill_wave };
}

static enum damacy_status
wave_input_rollback_desc(struct wave_pool* wp,
                         struct input_slot* slot,
                         const struct wave_desc* desc,
                         int* changed)
{
  render_job_rollback_wave(
    render_job_pool_get(wp->render_jobs, slot->render_job_idx), desc);
  wp->stats->waves_emitted--;
  wp->stats->chunks_dispatched -= slot->n_chunks;
  input_slot_release(slot);
  mark_changed(changed);
  return DAMACY_OK;
}

enum damacy_status
wave_input_rollback_slot(struct wave_pool* wp,
                         struct input_slot* slot,
                         int* changed)
{
  struct wave_desc desc = input_slot_wave_desc(slot);
  return wave_input_rollback_desc(wp, slot, &desc, changed);
}

int
wave_input_reservation_has_slot(const struct wave_input_reservation* r)
{
  return r && r->has_slot;
}

int
wave_input_reservation_slot_index(const struct wave_input_reservation* r)
{
  return r ? r->input_slot_idx : -1;
}

enum damacy_status
wave_input_reserve(struct wave_pool* wp,
                   uint16_t render_job_idx,
                   struct wave_input_reservation* out)
{
  *out = (struct wave_input_reservation){ 0 };
  struct render_job* job = render_job_pool_get(wp->render_jobs, render_job_idx);
  if (!job)
    return DAMACY_INVAL;
  if (!render_job_has_work(job))
    return DAMACY_OK;
  int input_slot_idx = input_slot_find_free(wp->slots, wp->n_slots);
  if (input_slot_idx < 0)
    return DAMACY_OK;
  struct input_slot* slot = &wp->slots[input_slot_idx];

  const struct wave_pack_limits limits = {
    .input_cap = slot->cap,
    .dev_decompressed_cap = wp->waves[0].dev_decompressed_cap,
    .max_chunks_per_wave = wp->max_chunks_per_wave,
  };
  struct wave_desc desc = { 0 };
  enum damacy_status s = wave_dispatcher_reserve(job,
                                                 render_job_idx,
                                                 &limits,
                                                 slot->store_reads,
                                                 wp->input->read_base(slot),
                                                 &desc);
  if (s != DAMACY_OK)
    return s;
  input_slot_begin_reservation(slot, &desc);
  wp->stats->waves_emitted++;
  wp->stats->chunks_dispatched += desc.n_chunks;
  wave_account(wp, slot, job->batch_id, &desc);

  out->has_slot = 1;
  out->input_slot_idx = input_slot_idx;
  out->n_reads = desc.n_reads;
  out->desc = desc;
  return DAMACY_OK;
}

struct store_submit_result
wave_input_submit(struct wave_pool* wp, const struct wave_input_reservation* t)
{
  if (!wave_input_reservation_has_slot(t) || t->n_reads == 0)
    return (struct store_submit_result){ .status = DAMACY_OK };
  struct input_slot* slot = &wp->slots[t->input_slot_idx];
  return wp->input->submit_reads(wp->store, slot->store_reads, t->n_reads);
}

enum damacy_status
wave_input_commit(struct wave_pool* wp,
                  struct wave_input_reservation* t,
                  struct store_submit_result submit,
                  int* changed)
{
  if (t->finalized) {
    log_error("wave: input_commit called twice on slot %d", t->input_slot_idx);
    return DAMACY_OK;
  }
  t->finalized = 1;
  if (!wave_input_reservation_has_slot(t))
    return DAMACY_OK;
  struct input_slot* slot = &wp->slots[t->input_slot_idx];
  if (submit.status != DAMACY_OK) {
    wave_input_rollback_desc(wp, slot, &t->desc, changed);
    return submit.status;
  }
  input_slot_commit_io(slot, submit.event);
  mark_changed(changed);
  return DAMACY_OK;
}
