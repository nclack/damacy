// Internal: filesystem store. Public callers use store.h only.
#pragma once

#include "store/store.h"
#include "store/store_internal.h"

#include "io_queue/io_queue.h"
#include "platform/platform.h"
#include "platform/platform_io.h"

#include <stddef.h>
#include <stdint.h>

struct lru;
struct lru_entry;
struct lru_stats;
struct pool;

struct store_fs
{
  struct store base;
  char* root; // owned
  struct io_queue* q;

  struct platform_mutex* cache_mu;
  struct lru* fd_cache;

  struct pool* job_pool;
};

platform_file*
store_fs_acquire(struct store_fs* fs,
                 const char* key,
                 struct lru_entry** pin_out);

void
store_fs_release(struct store_fs* fs, struct lru_entry* pin);

void
store_fs_stats_get(struct store_fs* fs, struct lru_stats* out);
