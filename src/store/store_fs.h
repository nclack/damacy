// Internal: filesystem store. Public callers use store.h only.
#pragma once

#include "store/store.h"

#include "io_queue/io_queue.h"
#include "platform/platform.h"
#include "platform/platform_io.h"

#include <stddef.h>
#include <stdint.h>

struct fs_cache_slot
{
  char* key;
  platform_file* file;
};

struct store_fs
{
  struct store base;
  char* root; // owned
  struct io_queue* q;

  // Open-file cache. Linear scan; small until we find we need better.
  struct platform_mutex* cache_mu;
  struct fs_cache_slot* slots;
  size_t n_slots;
  size_t cap_slots;
};

// Bridge for store_fs_gds.c — returns a cached handle (do not close).
platform_file*
store_fs_get_file_external(struct store_fs* fs, const char* key);
