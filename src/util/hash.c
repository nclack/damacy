#include "hash.h"

uint64_t
hash_fnv1a_init(void)
{
  return HASH_FNV_OFFSET_BASIS;
}

uint64_t
hash_fnv1a_step(uint64_t state, const void* bytes, size_t n_bytes)
{
  const uint8_t* byte = (const uint8_t*)bytes;
  for (size_t i = 0; i < n_bytes; ++i)
    state = (state ^ byte[i]) * HASH_FNV_PRIME;
  return state;
}

uint64_t
hash_fnv1a(const void* bytes, size_t n_bytes)
{
  return hash_fnv1a_step(hash_fnv1a_init(), bytes, n_bytes);
}

uint64_t
hash_fnv1a_str(const char* s)
{
  uint64_t state = hash_fnv1a_init();
  for (; *s; ++s)
    state = (state ^ (uint8_t)*s) * HASH_FNV_PRIME;
  return state;
}

uint64_t
hash_ptr(const void* p)
{
  uint64_t x = (uint64_t)(uintptr_t)p;
  x *= 0x9e3779b97f4a7c15ull;
  x ^= x >> 32;
  return x;
}

uint64_t
hash_combine(uint64_t a, uint64_t b)
{
  // Murmur-like finalizer applied to the combination.
  uint64_t mixed = a ^ (b + 0x9e3779b97f4a7c15ull + (a << 6) + (a >> 2));
  mixed ^= mixed >> 33;
  mixed *= 0xff51afd7ed558ccdull;
  mixed ^= mixed >> 33;
  mixed *= 0xc4ceb9fe1a85ec53ull;
  mixed ^= mixed >> 33;
  return mixed;
}
