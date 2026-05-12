// Internal config helpers: dtype mapping, validation, and the
// "0 means use the ceiling" knob resolvers.
#pragma once

#include "damacy.h"
#include "dtype/dtype.h"

#include <stdint.h>

uint32_t damacy_dtype_bpe(enum damacy_dtype dt);

// 1 if the assemble kernel will accept (src, dst). Sources are the
// common image dtypes; both bf16 and f32 destinations accept all of them.
int cast_path_supported(enum damacy_dtype dst, enum dtype src);

enum damacy_status validate_config(const struct damacy_config* cfg);

// 0 → compile-time ceiling.
uint8_t resolve_max_bpe(const struct damacy_config* cfg);

// 0 → 512 KB default; clamps to the kernel-array ceiling.
uint64_t resolve_max_chunk_uncompressed(const struct damacy_config* cfg);
