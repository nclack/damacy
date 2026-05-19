// Lightweight 64-bit hash helpers. FNV-1a for byte sequences plus a
// combiner so callers can build composite keys without pulling in a
// third-party hash library.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define HASH_FNV_OFFSET_BASIS 0xcbf29ce484222325ull
#define HASH_FNV_PRIME 0x00000100000001b3ull

  // Returns the FNV-1a offset basis (the "empty hash" seed).
  uint64_t hash_fnv1a_init(void);

  // Stream `n_bytes` of `bytes` into a running FNV-1a hash and return
  // the new state. Use with hash_fnv1a_init to incrementally hash data
  // that isn't contiguous in memory.
  uint64_t hash_fnv1a_step(uint64_t state, const void* bytes, size_t n_bytes);

  // Hash `n_bytes` of `bytes` in one call.
  uint64_t hash_fnv1a(const void* bytes, size_t n_bytes);

  // Hash a NUL-terminated string (the terminator is not included).
  uint64_t hash_fnv1a_str(const char* s);

  // Mix two hashes (or a hash and a counter) in a way that scrambles
  // low bits adequately for use as a hash table key.
  uint64_t hash_combine(uint64_t a, uint64_t b);

  // Caller must use stable/interned pointers; pointer-identity defines
  // equality.
  uint64_t hash_ptr(const void* p);

#ifdef __cplusplus
}
#endif
