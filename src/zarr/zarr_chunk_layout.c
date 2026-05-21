#include "zarr/zarr_chunk_layout.h"

#include "damacy_limits.h"
#include "store/store.h"
#include "util/prelude.h"
#include "zarr/zarr_metadata.h" // CODEC_BLOSC_ZSTD

#include <stdint.h>

#define BLOSC1_HEADER_BYTES 16u

static inline uint32_t
read_u32_le(const uint8_t* p)
{
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

int
zarr_chunk_layout_probe(struct store* s,
                        const char* shard_path,
                        uint64_t first_chunk_off,
                        uint32_t first_chunk_cbytes,
                        uint8_t codec_id,
                        uint32_t max_substreams_per_chunk,
                        struct chunk_layout* out)
{
  CHECK_SILENT(Fail, s);
  CHECK_SILENT(Fail, shard_path);
  CHECK_SILENT(Fail, out);

  // Only blosc-formatted codecs carry a per-chunk header.
  if (codec_id != (uint8_t)CODEC_BLOSC_ZSTD)
    return 1;
  if (first_chunk_cbytes < BLOSC1_HEADER_BYTES)
    return 1;

  uint8_t header[BLOSC1_HEADER_BYTES];
  struct store_read read = {
    .key = shard_path,
    .dst = header,
    .offset = first_chunk_off,
    .len = sizeof(header),
  };
  if (store_read_many(s, &read, 1))
    return 1;

  const uint8_t* p = header;
  const uint8_t flags = p[2];
  uint8_t typesize = p[3];
  uint32_t nbytes = read_u32_le(p + 4);
  uint32_t blocksize = read_u32_le(p + 8);
  // p+12 is cbytes; redundant with first_chunk_cbytes, skipped.

  if (blocksize == 0)
    return 1;
  if (nbytes > DAMACY_BLOSC_MAX_CHUNK_UNCOMPRESSED_BYTES)
    return 1;
  if (typesize == 0 || typesize > 8u)
    return 1;
  uint32_t nblocks = nbytes / blocksize + (nbytes % blocksize != 0u);
  if (nblocks > max_substreams_per_chunk)
    return 1;

  *out = (struct chunk_layout){
    .codec_id = codec_id,
    .typesize = typesize,
    .shuffle = (uint8_t)(flags & 0x01u),
    .bitshuffle = (uint8_t)((flags >> 2) & 0x01u),
    .memcpyed = (uint8_t)((flags >> 1) & 0x01u),
    .dont_split = (uint8_t)((flags >> 4) & 0x01u),
    .blocksize = blocksize,
    .nbytes = nbytes,
    .nblocks = nblocks,
  };
  return 0;

Fail:
  return 1;
}
