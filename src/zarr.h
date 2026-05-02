// Public sharded zarr v3 reader.
//
// v0 scope:
//   - zarr v3 array metadata only (zarr.json)
//   - sharded layout (sharding_indexed codec) — required
//   - zstd inner codec — required
//   - no LOD / multiscale (NGFF wraps this layer; v1)
//   - rank up to 8
//
// Lifetime: the reader borrows the store; the caller must keep the store
// alive for at least as long as the reader.
#pragma once

#include "dimension.h"
#include "dtype.h"
#include "store.h"
#include "types.codec.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define DAMACY_MAX_RANK 8

  struct zarr_reader;

  struct zarr_reader_config
  {
    struct store* store; // not owned
    const char* prefix;  // path to the array within the store; "" = root
  };

  struct zarr_array_info
  {
    enum dtype dtype;
    uint8_t rank;
    const struct dimension* dims; // length = rank; size + chunk_size +
                                  // chunks_per_shard populated
    struct codec_config codec;    // inner codec (zstd in v0)
  };

  // Open a reader. Reads zarr.json from <store_root>/<prefix>/zarr.json,
  // validates the layout, and caches the metadata. Returns NULL on
  // failure.
  struct zarr_reader* zarr_reader_open(const struct zarr_reader_config* cfg);

  void zarr_reader_close(struct zarr_reader* r);

  const struct zarr_array_info* zarr_reader_info(const struct zarr_reader* r);

  // Resolve a chunk coordinate to its location in the store. coord has
  // `rank` entries in chunk-grid units. Returns 0 on hit, non-zero if the
  // chunk is empty (not stored) or the coord is out of range.
  struct zarr_chunk_loc
  {
    const char* key; // shard key (interned in the reader; valid until
                     // zarr_reader_close)
    uint64_t offset; // byte offset within the shard
    size_t len;      // compressed bytes
  };

  int zarr_reader_locate(struct zarr_reader* r,
                         const int64_t* chunk_coord,
                         struct zarr_chunk_loc* out);

  // Largest compressed chunk over all shard indices loaded so far. Useful
  // for sizing staging buffers up front.
  size_t zarr_reader_chunk_max_compressed_bytes(const struct zarr_reader* r);

  // Uncompressed chunk size in bytes (chunk volume × bpe). Constant for the
  // whole array.
  size_t zarr_reader_chunk_uncompressed_bytes(const struct zarr_reader* r);

#ifdef __cplusplus
}
#endif
