// Lightweight 64-bit hash helpers. FNV-1a for byte sequences plus a
// combiner so callers can build composite keys without pulling in a
// third-party hash library.
#pragma once

#include <stddef.h>
#include <stdint.h>

#define HASH_FNV_OFFSET_BASIS 0xcbf29ce484222325ull
#define HASH_FNV_PRIME 0x00000100000001b3ull

static inline uint64_t
hash_fnv1a_init(void)
{
  return HASH_FNV_OFFSET_BASIS;
}

static inline uint64_t
hash_fnv1a_step(uint64_t h, const void* bytes, size_t n)
{
  const uint8_t* p = (const uint8_t*)bytes;
  for (size_t i = 0; i < n; ++i)
    h = (h ^ p[i]) * HASH_FNV_PRIME;
  return h;
}

static inline uint64_t
hash_fnv1a(const void* bytes, size_t n)
{
  return hash_fnv1a_step(hash_fnv1a_init(), bytes, n);
}

static inline uint64_t
hash_fnv1a_str(const char* s)
{
  uint64_t h = hash_fnv1a_init();
  for (; *s; ++s)
    h = (h ^ (uint8_t)*s) * HASH_FNV_PRIME;
  return h;
}

// Mix two hashes (or a hash and a counter) in a way that scrambles low
// bits adequately for use as a hash table key.
static inline uint64_t
hash_combine(uint64_t a, uint64_t b)
{
  // Murmur-like finalizer applied to the combination.
  uint64_t h = a ^ (b + 0x9e3779b97f4a7c15ull + (a << 6) + (a >> 2));
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdull;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53ull;
  h ^= h >> 33;
  return h;
}
