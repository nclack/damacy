// Internal config helpers: dtype mapping, validation, and the
// "0 means use the ceiling" knob resolvers.
#pragma once

#include "damacy.h"
#include "dtype/dtype.h"

#include <stdint.h>

uint32_t
damacy_dtype_bpe(enum damacy_dtype dt);

// 1 if the assemble kernel will accept (src, dst). Sources are the
// common image dtypes; both bf16 and f32 destinations accept all of them.
int
cast_path_supported(enum damacy_dtype dst, enum dtype src);

enum damacy_status
validate_config(const struct damacy_config* cfg);

// 0 → 512 KB default; clamps to the kernel-array ceiling.
uint64_t
resolve_max_chunk_uncompressed(const struct damacy_config* cfg);

// 0 → DAMACY_DEFAULT_MAX_GPU_MEMORY_BYTES (~1 GB). Primary GPU budget
// knob — every other GPU-allocation sizing derives from this.
uint64_t
resolve_max_gpu_memory(const struct damacy_config* cfg);

// 0 → DAMACY_DEFAULT_HOST_BUFFER_WAVES (3); clamped to
// [DAMACY_N_WAVES, DAMACY_MAX_HOST_BUFFER_WAVES].
uint8_t
resolve_host_buffer_waves(const struct damacy_config* cfg);

// Resolves the enable_gds flag: returns 1 when either cfg->enable_gds
// is non-zero or the DAMACY_GDS_ENABLE environment variable is set to
// "1". 0 otherwise. damacy_create rejects with DAMACY_INVAL when this
// resolves to 1 and libcufile.so.0 is not loadable / cuFileDriverOpen
// fails inside store_fs_create.
uint8_t
resolve_enable_gds(const struct damacy_config* cfg);
