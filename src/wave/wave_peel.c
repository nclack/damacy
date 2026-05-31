#include "wave_peel.h"

#include "log/log.h"
#include "wave_pool.h"

static void
mark_changed(int* changed)
{
  if (changed)
    *changed = 1;
}

static void
slot_rollback_peel(struct wave_pool* wp,
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
}

struct wave_pool_peel_ticket
wave_pool_peel_reserve(struct wave_pool* wp,
                       uint16_t render_job_idx,
                       enum damacy_status* err)
{
  *err = DAMACY_OK;
  struct wave_pool_peel_ticket t = { .input_slot_idx = -1,
                                     .n_reads = 0,
                                     .committed = 0 };
  struct render_job* job = render_job_pool_get(wp->render_jobs, render_job_idx);
  if (!job) {
    *err = DAMACY_INVAL;
    return t;
  }
  if (!render_job_has_work(job))
    return t;
  int input_slot_idx = input_slot_find_free(wp->slots, wp->n_slots);
  if (input_slot_idx < 0)
    return t;
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
  if (s != DAMACY_OK) {
    *err = s;
    return t;
  }
  input_slot_begin_peel(slot, &desc);
  wp->stats->waves_emitted++;
  wp->stats->chunks_dispatched += desc.n_chunks;

  t.input_slot_idx = input_slot_idx;
  t.n_reads = desc.n_reads;
  t.desc = desc;
  return t;
}

struct store_event
wave_pool_peel_submit(struct wave_pool* wp,
                      const struct wave_pool_peel_ticket* t)
{
  if (t->input_slot_idx < 0 || t->n_reads == 0)
    return (struct store_event){ .seq = 0 };
  struct input_slot* slot = &wp->slots[t->input_slot_idx];
  return wp->input->submit_reads(wp->store, slot->store_reads, t->n_reads);
}

enum damacy_status
wave_pool_peel_commit(struct wave_pool* wp,
                      struct wave_pool_peel_ticket* t,
                      struct store_event ev,
                      int* changed)
{
  if (t->committed) {
    log_error("wave: peel_commit called twice on slot %d", t->input_slot_idx);
    return DAMACY_OK;
  }
  t->committed = 1;
  if (t->input_slot_idx < 0)
    return DAMACY_OK;
  struct input_slot* slot = &wp->slots[t->input_slot_idx];
  if (t->n_reads > 0 && ev.seq == 0) {
    slot_rollback_peel(wp, slot, &t->desc, changed);
    return DAMACY_IO;
  }
  input_slot_commit_io(slot, ev);
  mark_changed(changed);
  return DAMACY_OK;
}
