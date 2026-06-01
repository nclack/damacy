#include "wave_input.h"

#include "log/log.h"
#include "wave_pool.h"

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

  out->active = 1;
  out->input_slot_idx = input_slot_idx;
  out->n_reads = desc.n_reads;
  out->desc = desc;
  return DAMACY_OK;
}

struct store_submit_result
wave_input_submit(struct wave_pool* wp, const struct wave_input_reservation* t)
{
  if (!t->active || t->n_reads == 0)
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
  if (t->committed) {
    log_error("wave: input_commit called twice on slot %d", t->input_slot_idx);
    return DAMACY_OK;
  }
  t->committed = 1;
  if (!t->active)
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
