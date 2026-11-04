#include "zarr/zarr_chunk_layout.h"

#include "damacy_limits.h"
#include "store/store.h"
#include "util/prelude.h"
#include "zarr/zarr_metadata.h"

#include <stdint.h>

#define BLOSC1_HEADER_BYTES 16u

static inline uint32_t
read_u32_le(const uint8_t* p)
{
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

int
zarr_chunk_layout_parse_header(const void* header,
                               uint32_t first_chunk_cbytes,
                               uint8_t codec_id,
                               uint32_t max_substreams_per_chunk,
                               struct chunk_layout* out)
{
  CHECK_SILENT(Fail, header);
  CHECK_SILENT(Fail, out);

  // Only blosc-formatted codecs carry a per-chunk header.
  if (codec_id != (uint8_t)CODEC_BLOSC_ZSTD &&
      codec_id != (uint8_t)CODEC_BLOSC_LZ4)
    return 1;
  if (first_chunk_cbytes < BLOSC1_HEADER_BYTES)
    return 1;

  const uint8_t* p = (const uint8_t*)header;
  const uint8_t flags = p[2];
  uint8_t typesize = p[3];
  uint32_t nbytes = read_u32_le(p + 4);
  uint32_t blocksize = read_u32_le(p + 8);
  if (p[0] < 1 || p[0] > 2 || read_u32_le(p + 12) != first_chunk_cbytes)
    return 1;
  if (!nbytes || nbytes > DAMACY_BLOSC_MAX_CHUNK_UNCOMPRESSED_BYTES ||
      !blocksize || blocksize > nbytes || !typesize || typesize > 8u ||
      nbytes % typesize || blocksize % typesize)
    return 1;
  const uint8_t memcpyed = (flags >> 1) & 1u;
  const uint8_t compformat = codec_id == CODEC_BLOSC_LZ4 ? 1u : 4u;
  if (memcpyed) {
    if (first_chunk_cbytes - BLOSC1_HEADER_BYTES != nbytes)
      return 1;
  } else if (p[1] != 1 || (flags >> 5) != compformat || (flags & 8u)) {
    return 1;
  }
  const uint32_t nblocks = nbytes / blocksize + (nbytes % blocksize != 0u);
  const uint32_t nsplits =
    !(flags & 0x10u) && blocksize / typesize >= 128u ? typesize : 1u;
  const uint32_t nsubstreams =
    (nbytes / blocksize) * nsplits + (nbytes % blocksize != 0u);
  if (nsubstreams > max_substreams_per_chunk ||
      (!memcpyed && nblocks > (first_chunk_cbytes - BLOSC1_HEADER_BYTES) / 4u))
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
    .nsubstreams = nsubstreams,
  };
  return 0;

Fail:
  return 1;
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
  if (codec_id != (uint8_t)CODEC_BLOSC_ZSTD &&
      codec_id != (uint8_t)CODEC_BLOSC_LZ4)
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

  return zarr_chunk_layout_parse_header(
    header, first_chunk_cbytes, codec_id, max_substreams_per_chunk, out);

Fail:
  return 1;
}
