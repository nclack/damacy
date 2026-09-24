#pragma once

#include "damacy_pipeline.h"
#include "planner/plan.h"
#include "platform/platform.h"
#include "store/store.h"

#include <stdatomic.h>

struct damacy_reader
{
  struct store* store;
  uint32_t max_inflight_reads;
};

struct damacy_metadata_reader
{
  uint32_t concurrency;
  struct damacy_latency_model latency;
};

struct damacy_metadata
{
  struct damacy_metadata_reader reader;
  struct damacy_metadata_cache_config cache;
};

struct damacy_planner_ops
{
  enum damacy_status (*start)(struct damacy_planner*,
                              const struct damacy_batch_spec*,
                              const struct damacy_queue_limits*);
  struct damacy_push_result (*push)(struct damacy_planner*,
                                    struct damacy_sample_slice);
  enum damacy_status (*next)(struct damacy_planner*, struct prepared_plan**);
  uint64_t (*pending)(const struct damacy_planner*);
  void (*stats)(struct damacy_planner*, struct damacy_stats*);
  void (*reset_stats)(struct damacy_planner*);
  void (*stop)(struct damacy_planner*);
  void (*destroy)(struct damacy_planner*);
};

struct damacy_planner
{
  const struct damacy_planner_ops* ops;
  _Atomic int active;
};

struct damacy_executor_ops
{
  enum damacy_status (*enter_thread)(struct damacy_executor*);
  void (*leave_thread)(struct damacy_executor*);
  enum damacy_status (*start)(struct damacy_executor*,
                              const struct damacy_batch_spec*,
                              struct damacy_stats*);
  enum damacy_status (*submit)(struct damacy_executor*,
                               struct prepared_plan*,
                               uint64_t);
  enum damacy_status (*step)(struct damacy_executor*, int*);
  enum damacy_status (*take)(struct damacy_executor*, struct damacy_batch**);
  enum damacy_status (*wait_event)(struct damacy_executor*, void*);
  int (*busy)(const struct damacy_executor*);
  void (*stats)(struct damacy_executor*, struct damacy_stats*);
  void (*stop)(struct damacy_executor*);
  void (*destroy)(struct damacy_executor*);
};

struct damacy_executor
{
  const struct damacy_executor_ops* ops;
  _Atomic int active;
  int device_type;
  int device_id;
};

struct damacy_buffer
{
  _Atomic uint32_t references;
  void* data;
  uint64_t nbytes;
  void* ready_stream;
  int device_type;
  int device_id;
  void (*destroy)(struct damacy_buffer*);
  enum damacy_status (*wait_event)(struct damacy_buffer*, void*);
  void* context;
};

struct damacy_batch
{
  _Atomic uint32_t references;
  struct damacy_buffer* buffer;
  struct damacy_batch_info info;
  const struct damacy* owner;
};

enum damacy_status
batch_spec_layout(const struct damacy_batch_spec* spec,
                  int64_t* shape,
                  int64_t* strides,
                  uint64_t* bytes);
struct damacy_batch*
batch_create(struct damacy_buffer* buffer,
             const struct damacy_batch_spec* output,
             uint64_t batch_id);
void
buffer_retain(struct damacy_buffer* buffer);
void
buffer_release(struct damacy_buffer* buffer);
int
buffer_available(const struct damacy_buffer* buffer);
