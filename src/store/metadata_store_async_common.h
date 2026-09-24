// Backend-author header: metrics, injected latency, and error mapping shared
// by the io_uring and POSIX metadata backends.
#pragma once

#include "store/metadata_store_async.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>

enum op_kind
{
  OP_STATX,
  OP_OPEN,
  OP_READ,
  OP_CLOSE,
};

enum latency_op_kind
{
  LATENCY_OP_STAT,
  LATENCY_OP_SUBMIT,
};

struct metadata_metrics
{
  struct
  {
    _Atomic uint64_t ops;
    _Atomic uint64_t stat_ops;
    _Atomic uint64_t submit_ops;
    _Atomic uint64_t active;
    _Atomic uint64_t max_active;
    _Atomic uint64_t total_sleep_ns;
    _Atomic uint64_t max_sleep_ns;
  } injector;

  struct
  {
    _Atomic uint64_t jobs;
    _Atomic uint64_t active;
    _Atomic uint64_t max_active;
  } read;

  struct
  {
    _Atomic uint64_t count[METADATA_OP_LATENCY_NKINDS];
    _Atomic uint64_t sum_ns[METADATA_OP_LATENCY_NKINDS];
    _Atomic uint64_t max_ns[METADATA_OP_LATENCY_NKINDS];
    _Atomic uint64_t buckets[METADATA_OP_LATENCY_NKINDS]
                            [METADATA_OP_LATENCY_NBUCKETS];
  } op_latency;
};

struct metadata_store_common
{
  struct damacy_latency_model latency;
  int latency_enabled;
  uint64_t rng_state;
  pthread_mutex_t rng_lock;
  struct metadata_metrics metrics;
};

// Each backend defines this; the stats functions reach the metrics through it.
struct metadata_store_common*
metadata_store_async_common(struct metadata_store_async* s);

int
metadata_store_common_init(struct metadata_store_common* c,
                           const struct damacy_latency_model* latency);
void
metadata_store_common_destroy(struct metadata_store_common* c);

uint64_t
metadata_monotonic_ns(void);
void
metadata_record_op_latency(struct metadata_store_common* c,
                           enum op_kind kind,
                           uint64_t submit_ts);
void
metadata_inject_latency(struct metadata_store_common* c,
                        enum latency_op_kind kind);
void
metadata_read_active_begin(struct metadata_store_common* c);
void
metadata_read_active_end(struct metadata_store_common* c);

enum damacy_status
metadata_status_from_errno(int err);
enum store_stat_result
metadata_stat_status_from_errno(int err);
