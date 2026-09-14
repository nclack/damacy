#pragma once

#include "pipeline/components.h"
#include "scheduler/scheduler.h"

struct damacy
{
  struct damacy_batch_spec output;
  struct damacy_queue_limits queues;
  struct damacy_planner* planner;
  struct damacy_executor* executor;
  struct prepared_plan** plans;
  uint32_t plan_head;
  uint32_t plan_count;
  uint64_t next_batch_id;
  enum damacy_status failed_status;
  struct damacy_stats stats;
  struct scheduler* sched;
  uint32_t pop_calls;
  _Atomic int stopping;
  _Atomic int stopped;
  int device;
  int owns_components;
  struct damacy_reader* owned_reader;
  struct damacy_metadata_reader* owned_metadata_reader;
  struct damacy_metadata* owned_metadata;
};

int
damacy_scheduler_step(void* arg);
void
damacy_scheduler_enter(void* arg);
void
damacy_scheduler_leave(void* arg);
enum damacy_status
pipeline_prepare(struct damacy* self, int* changed);
