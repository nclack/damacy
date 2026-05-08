#include "zarr/zarr_shard_index.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#define countof(arr) (sizeof(arr) / sizeof((arr)[0]))

static void
try_parse(const uint8_t* data, size_t size, size_t n_entries)
{
  // malloc(0) is implementation-defined.
  size_t alloc_n = n_entries ? n_entries : 1;
  struct zarr_shard_entry* entries =
    (struct zarr_shard_entry*)malloc(alloc_n * sizeof(*entries));
  if (!entries)
    return;
  (void)zarr_shard_index_parse(data, size, n_entries, entries);
  free(entries);
}

int
LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
  // Honest n_entries derived the way the cache layer does.
  if (size >= 4 && (size - 4) % 16 == 0) {
    try_parse(data, size, (size - 4) / 16);
  }

  // Adversarial n_entries drives the size-mismatch reject branch.
  static const size_t mismatched[] = { 0, 1, 2, 7, 64, 1024 };
  for (size_t i = 0; i < countof(mismatched); ++i) {
    try_parse(data, size, mismatched[i]);
  }

  // NULL-buf branch isn't reachable through try_parse.
  if (size == 0) {
    (void)zarr_shard_index_parse(NULL, 0, 0, NULL);
  }

  return 0;
}
