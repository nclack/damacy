// Two batch slots in a state machine. Each slot owns the planner output
// (read_ops, chunk_plans, sample_plans + device mirror) plus a lazily-
// allocated device output tensor. The pool also caches the shared
// shape/strides established by the first push.
//
// Lifetime: batch_slot_init at create-time; batch_pool_compute_layout +
// batch_pool_alloc_dev at the first push (idempotent); batch_pool_destroy
// at destroy-time. cuda_skip=1 leaks GPU resources but releases the host
// heap (used when the CUDA context is no longer valid).
#pragma once

#include "damacy.h"
#include "damacy_limits.h"
#include "util/path_intern.h"

#include <stdint.h>

enum batch_slot_state
{
  BATCH_FREE = 0,
  BATCH_OPEN,     // planned prefix, waiting for more samples or flush close
  BATCH_PLANNING, // reserve has drained samples; run/commit pending
  BATCH_FILLING,  // planner has emitted; chunks may or may not be dispatched
  BATCH_READY,    // chunks_remaining == 0; awaiting pop
  BATCH_HELD,     // user holds the handle
};

// Forward decls; full types come from planner/planner.h. We reference
// them only by pointer here so the header doesn't depend on planner.
struct read_op;
struct chunk_plan;
struct sample_plan;
struct read_op_group;
struct planner_sample;

struct damacy_batch_slot
{
  enum batch_slot_state state;
  uint64_t batch_id;
  uint64_t sample_seq_begin;
  uint32_t n_samples; // shape[0]: number of complete samples
  void* dev_ptr;      // device output tensor (allocated lazily)

  struct read_op* read_ops;             // size DAMACY_MAX_CHUNKS_PER_BATCH
  struct chunk_plan* chunk_plans;       // size DAMACY_MAX_CHUNKS_PER_BATCH
  struct read_op_group* read_op_groups; // size DAMACY_MAX_CHUNKS_PER_BATCH
  uint32_t n_read_op_groups;
  struct sample_plan* sample_plans; // size cfg.samples_per_batch
  // Reset before each plan into the slot; bounds the working set to
  // distinct shard paths in one batch.
  struct path_intern paths;
  struct planner_sample* stage_samples; // size cfg.samples_per_batch
  void* d_sample_plans;    // device mirror, uploaded once per batch
  uint32_t n_sample_plans; // == n_samples on success
  uint32_t n_chunks;
  uint32_t n_chunks_to_load;    // non-fill chunks (filter survivors)
  uint32_t n_loads_issued;      // real read_ops after coalesce
  uint32_t n_chunks_dispatched; // 0 .. n_chunks; chunks given to a wave
  uint32_t n_groups_dispatched; // 0 .. n_read_op_groups; groups handed to waves
  int32_t chunks_remaining;     // n_chunks - chunks completed via waves

  // Set by damacy_release_event when a deferred-release wait has been
  // queued on stream_post. plan_commit's degenerate (zero-chunk) path
  // reads this and host-syncs stream_post before its sync cuMemsetD8;
  // the normal reuse path is already gated by the cuStreamWaitEvent on
  // stream_post itself.
  int deferred_release_pending;
  int planning_close_batch;
};

struct damacy_batch_pool
{
  struct damacy_batch_slot slots[2];
  uint64_t n_bytes;                     // size of one slot's output
  uint8_t rank;                         // includes leading N axis
  int64_t shape[DAMACY_MAX_RANK + 1];   // [samples_per_batch, ...sample_axes]
  int64_t strides[DAMACY_MAX_RANK + 1]; // row-major elements
  int layout_set;                       // shape/strides/n_bytes computed
  int allocated;                        // dev_ptrs alloc'd (implies layout_set)
};

// Returns 0 on success, non-zero on alloc failure.
int
batch_slot_init(struct damacy_batch_slot* slot, uint32_t samples_per_batch_cap);
void
batch_slot_destroy(struct damacy_batch_slot* slot, int cuda_skip);
void
batch_pool_destroy(struct damacy_batch_pool* pool, int cuda_skip);

// Establishes shape/strides/n_bytes from cfg->sample_shape. No GPU
// touch. Idempotent on the same input — sets pool->layout_set on first
// success and short-circuits subsequent calls. sample_rank must be in
// [1, DAMACY_MAX_RANK] and every sample_shape[d] must be > 0; otherwise
// DAMACY_INVAL. The caller checks the budget against pool->n_bytes and
// then calls batch_pool_alloc_dev.
enum damacy_status
batch_pool_compute_layout(struct damacy_batch_pool* pool,
                          const int64_t* sample_shape,
                          uint8_t sample_rank,
                          uint32_t samples_per_batch,
                          uint32_t bpe);

// Allocates dev_ptr for both slots (size pool->n_bytes each). Requires
// pool->layout_set. Idempotent — sets pool->allocated on first success.
// Caller bumps gpu_bytes_committed by 2 × pool->n_bytes on success.
enum damacy_status
batch_pool_alloc_dev(struct damacy_batch_pool* pool);

// All return -1 / 0 if the predicate is unmet. find_oldest_*
// scans by lowest batch_id. find_filling_slot_with_work additionally
// requires n_chunks_dispatched < n_chunks (used by the wave scheduler).
int
find_free_batch_slot(const struct damacy_batch_pool* pool);
int
find_open_batch_slot(const struct damacy_batch_pool* pool);
int
find_oldest_ready_slot(const struct damacy_batch_pool* pool);
int
find_oldest_filling_slot(const struct damacy_batch_pool* pool);
int
find_filling_slot_with_work(const struct damacy_batch_pool* pool);
int
any_batch_in_flight(const struct damacy_batch_pool* pool);
// True if any slot is BATCH_PLANNING. Used by damacy_flush to wait for
// a plan that has reserved a slot (drained samples) but not yet
// committed (so find_oldest_filling_slot doesn't yet see it).
int
any_batch_planning(const struct damacy_batch_pool* pool);

// Subtract `n_consumed` chunks from the slot's outstanding work. If
// the slot was FILLING and the count reaches 0, transitions it to
// READY. Called by the orchestrator when a wave's chunks have
// retired. No-op on non-FILLING/READY slots.
void
batch_slot_consume_chunks(struct damacy_batch_slot* slot, uint32_t n_consumed);
