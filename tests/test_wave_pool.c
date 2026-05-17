// Unit tests for the host-side parts of src/wave/wave_pool.c.
//
//   test_peel_commit_rollback_once — a submit-failure rollback decrements
//     stats once; a second commit on the same ticket is a no-op so the
//     counters don't underflow.
//   test_peel_commit_success_then_recommit — success path transitions the
//     slot to SLOT_IO; a second commit is a no-op.
//
// We don't exercise wave_pool_init / wave_pool_peel_reserve here: those
// need CUDA. peel_commit is pure host code touching wp->stats,
// wp->pool->slots, and a host_slab_slot — all constructible on the stack.

#include "batch_pool/batch_pool.h"
#include "damacy.h"
#include "damacy_log.h"
#include "expect.h"
#include "store/store.h"
#include "wave/host_slab.h"
#include "wave/wave_pool.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Construct just enough wave_pool state for peel_commit to run. The slot
// is wired as if peel_reserve had just been called: SLOT_PEELING, n_chunks
// set, batch_pool_slot points at slot 0 of the batch pool. Counters are
// pre-incremented to reflect a completed reserve.
static void
setup_post_reserve(struct wave_pool* wp,
                   struct damacy_batch_pool* batch_pool,
                   struct damacy_stats* stats,
                   struct host_slab_slot* slot,
                   uint32_t n_chunks)
{
  memset(wp, 0, sizeof(*wp));
  memset(batch_pool, 0, sizeof(*batch_pool));
  memset(stats, 0, sizeof(*stats));
  memset(slot, 0, sizeof(*slot));

  wp->pool = batch_pool;
  wp->stats = stats;
  wp->slots[0] = *slot; // placeholder; we use the pointer below

  // Wire the fields peel_commit actually reads.
  wp->slots[0].state = SLOT_PEELING;
  wp->slots[0].batch_pool_slot = 0;
  wp->slots[0].n_chunks = n_chunks;

  batch_pool->slots[0].n_chunks_dispatched = n_chunks;
  stats->waves_emitted = 1;
  stats->chunks_dispatched = n_chunks;
}

static int
test_peel_commit_rollback_once(void)
{
  struct wave_pool wp;
  struct damacy_batch_pool batch_pool;
  struct damacy_stats stats;
  struct host_slab_slot slot;
  const uint32_t n_chunks = 3;
  setup_post_reserve(&wp, &batch_pool, &stats, &slot, n_chunks);

  // n_reads > 0 + ev.seq == 0 → submit-failure rollback path.
  struct wave_pool_peel_ticket t = { .slot_idx = 0,
                                     .n_reads = 2,
                                     .consumed = 0 };
  struct store_event ev = { .seq = 0 };

  EXPECT(wave_pool_peel_commit(&wp, &t, ev) == DAMACY_IO);
  EXPECT(t.consumed == 1);
  EXPECT(stats.waves_emitted == 0);
  EXPECT(stats.chunks_dispatched == 0);
  EXPECT(batch_pool.slots[0].n_chunks_dispatched == 0);
  EXPECT(wp.slots[0].state == SLOT_FREE);

  // Second call must NOT decrement again — that would underflow.
  // Re-entry intentionally trips a log_error; quiet the sink across it.
  damacy_log_set_quiet(1);
  EXPECT(wave_pool_peel_commit(&wp, &t, ev) == DAMACY_OK);
  damacy_log_set_quiet(0);
  EXPECT(stats.waves_emitted == 0);
  EXPECT(stats.chunks_dispatched == 0);
  EXPECT(batch_pool.slots[0].n_chunks_dispatched == 0);
  return 0;
}

static int
test_peel_commit_success_then_recommit(void)
{
  struct wave_pool wp;
  struct damacy_batch_pool batch_pool;
  struct damacy_stats stats;
  struct host_slab_slot slot;
  const uint32_t n_chunks = 4;
  setup_post_reserve(&wp, &batch_pool, &stats, &slot, n_chunks);

  // Successful submit: ev.seq != 0 → slot advances to SLOT_IO.
  struct wave_pool_peel_ticket t = { .slot_idx = 0,
                                     .n_reads = 3,
                                     .consumed = 0 };
  struct store_event ev = { .seq = 42 };

  EXPECT(wave_pool_peel_commit(&wp, &t, ev) == DAMACY_OK);
  EXPECT(t.consumed == 1);
  EXPECT(wp.slots[0].state == SLOT_IO);
  EXPECT(wp.slots[0].io_event.seq == 42);
  // Success path leaves counters as the reserve set them.
  EXPECT(stats.waves_emitted == 1);
  EXPECT(stats.chunks_dispatched == n_chunks);

  // Second call: a stray re-commit (e.g. error retry) must not touch state.
  // Re-entry intentionally trips a log_error; quiet the sink across it.
  struct store_event ev2 = { .seq = 0 };
  damacy_log_set_quiet(1);
  EXPECT(wave_pool_peel_commit(&wp, &t, ev2) == DAMACY_OK);
  damacy_log_set_quiet(0);
  EXPECT(wp.slots[0].state == SLOT_IO);
  EXPECT(stats.waves_emitted == 1);
  EXPECT(stats.chunks_dispatched == n_chunks);
  return 0;
}

// Submit-failure rollback must restore n_groups_dispatched from the
// snapshot the ticket captured at reserve time — without it the next
// peel would skip past the groups this wave never actually issued.
static int
test_peel_commit_rolls_back_groups(void)
{
  struct wave_pool wp;
  struct damacy_batch_pool batch_pool;
  struct damacy_stats stats;
  struct host_slab_slot slot;
  const uint32_t n_chunks = 5;
  setup_post_reserve(&wp, &batch_pool, &stats, &slot, n_chunks);

  // Simulate reserve having advanced from group 7 to group 9.
  batch_pool.slots[0].n_groups_dispatched = 9;

  struct wave_pool_peel_ticket t = {
    .slot_idx = 0, .n_reads = 1, .prev_n_groups_dispatched = 7, .consumed = 0
  };
  struct store_event ev = { .seq = 0 };

  EXPECT(wave_pool_peel_commit(&wp, &t, ev) == DAMACY_IO);
  EXPECT(batch_pool.slots[0].n_groups_dispatched == 7);
  return 0;
}

static int
test_peel_commit_noop_ticket(void)
{
  // slot_idx < 0 means peel_reserve found no work or no free slab.
  // commit must short-circuit before touching anything.
  struct wave_pool wp;
  struct damacy_batch_pool batch_pool;
  struct damacy_stats stats;
  memset(&wp, 0, sizeof(wp));
  memset(&batch_pool, 0, sizeof(batch_pool));
  memset(&stats, 0, sizeof(stats));
  wp.pool = &batch_pool;
  wp.stats = &stats;

  struct wave_pool_peel_ticket t = { .slot_idx = -1,
                                     .n_reads = 0,
                                     .consumed = 0 };
  struct store_event ev = { .seq = 0 };
  EXPECT(wave_pool_peel_commit(&wp, &t, ev) == DAMACY_OK);
  EXPECT(stats.waves_emitted == 0);
  EXPECT(stats.chunks_dispatched == 0);
  return 0;
}

int
main(void)
{
  RUN(test_peel_commit_rollback_once);
  RUN(test_peel_commit_success_then_recommit);
  RUN(test_peel_commit_rolls_back_groups);
  RUN(test_peel_commit_noop_ticket);
  printf("all wave_pool tests passed\n");
  return 0;
}
