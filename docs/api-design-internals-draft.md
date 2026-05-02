# Internal architecture for damacy

> Draft produced 2026-04-29 by the `internals` agent during the API design
> walkthrough. Parked while the C-language kvikio benchmark proceeds. This
> represents one independent angle on the architecture; some open questions and
> naming have evolved since (see devlog and project memory).

## Components (layered, top → bottom)

```
┌─────────────────────────────────────────────────────┐
│ SampleStream  (public-surface-facing iterator)      │
└────────────┬────────────────────────────────────────┘
             │ batch request: N × (uri, aabb, mip)
┌────────────▼────────────────────────────────────────┐
│ BatchScheduler   — owns lookahead window & streams  │
└────────────┬────────────────────────────────────────┘
             │ resolved samples
┌────────────▼────────────────────────────────────────┐
│ ChunkPlanner   — (uri,aabb,mip) → [ChunkRef…]       │
└────────────┬────────────────────────────────────────┘
             │ ChunkRefs
┌────────────▼─────────────┐    ┌────────────────────┐
│ ChunkCache (compressed)  │◄──►│ ZarrMetadataCache  │
│   key=(uri,mip,coord)    │    │ .zarray/codec/etc  │
└────────────┬─────────────┘    └────────────────────┘
             │ misses
┌────────────▼────────────────────────────────────────┐
│ ChunkLoader    — kvikio raw_read_async → dev bytes  │
│                  on  reader_stream                  │
└────────────┬────────────────────────────────────────┘
             │ compressed dev buffers + cuda events
┌────────────▼────────────────────────────────────────┐
│ ChunkDecompressor — Blosc-parse + nvCOMP batch      │
│                     on  decoder_stream              │
└────────────┬────────────────────────────────────────┘
             │ decompressed chunk tiles (device)
┌────────────▼────────────────────────────────────────┐
│ BatchAssembler — gather kernel → output tensor      │
│                  on  assembler_stream               │
└─────────────────────────────────────────────────────┘
```

## Per-component contract

| Component | Responsibility | In | Out | Where |
|---|---|---|---|---|
| `SampleStream` | Public iterator; hands batches to caller | user calls | `Batch` (device tensor + meta) | CPU |
| `BatchScheduler` | Owns lookahead queue; issues planning N+K ahead; reclaims | sample specs | scheduled work | CPU |
| `ZarrMetadataCache` | Parses `.zarray`/`.zgroup`; caches chunk shape, dtype, codec, compressor params | uri | `ArrayMeta` | CPU |
| `ChunkPlanner` | Convert `(aabb,mip)` to chunk grid coords + intra-chunk slice | sample + meta | `vector<ChunkRef>` + `vector<SliceSpec>` | CPU |
| `ChunkCache` | LRU over **compressed** chunk bytes on device | `ChunkRef` | dev ptr + size or miss | CPU index, GPU storage |
| `ChunkLoader` | Issue `raw_read_async` for misses, stream-ordered into device | misses | dev compressed + event | reader_stream |
| `ChunkDecompressor` | Blosc-header parse on host; batched nvCOMP decode | compressed batch | dev decompressed tiles | decoder_stream |
| `BatchAssembler` | Single gather kernel: stitch tiles into `[N,T,C,Z,Y,X]` | tiles + SliceSpecs | output tensor | assembler_stream |

## Data flow for one batch

1. `SampleStream::next()` returns the batch promised K steps ago; scheduler immediately schedules batch `n+K`.
2. For each sample, `ChunkPlanner` computes the integer chunk-grid AABB at the chosen mip and the per-chunk source/dest slice spec. Pure integer math; no I/O.
3. The union of `ChunkRef`s is deduplicated against `ChunkCache`. Hits stay; misses become a read plan.
4. `ChunkLoader` enqueues all misses to `reader_stream` via `raw_read_async`. It records a CUDA event when the batch's reads complete.
5. `decoder_stream` waits on that event, then `ChunkDecompressor` issues `nvCOMP*GetTempSizeAsync → GetDecompressSizeAsync → DecompressAsync` over the whole batch's chunks (one launch per codec).
6. `assembler_stream` waits on the decoder event, then launches the gather kernel which writes each tile's intersection into the appropriate sample/region of the output.
7. Output tensor + completion event handed back to `SampleStream` as the realized batch.

## Position statements

- **Coalescing: skip in v1.** kvikio's thread-pool fanout already saturates NVMe queues when you have many in-flight `pread`s, and zarr chunks are often non-adjacent on disk anyway. Add only if profiling shows reader-stream idle.
- **Cache layer: compressed-only.** Decompressed tiles for higher-dim NGFF blow up VRAM 5–20×. Compressed cache lets us hold an order of magnitude more chunks; nvCOMP batch decode is cheap enough to redo on hit. Reserve a "hot decompressed" cache as a future optimization, not v1.
- **Lookahead depth: 2 (i.e., batch+1 in flight).** Three streams (reader/decoder/assembler) already give double-buffering; deeper queues mostly add latency and VRAM pressure. Configurable.
- **Single sample / single source / single mip** stays as a hard precondition of `ChunkPlanner`; multi-source sampling becomes a higher layer that fans out to N planners.
- **Error semantics: fail the whole batch, surface the failing `(uri, ChunkRef)`.** Partial batches are a footgun for training loops (silent shape changes, biased sampling). Loader catches I/O errors, cancels pending decodes for that batch, propagates a typed exception.
- **Streams: 3 fixed per `SampleStream` instance**, not per batch. Events synchronize across them; no host-side waits in the steady state.

## Open questions

- Does the public surface want per-sample dtype conversion / normalization, or is that the caller's responsibility post-batch? Affects whether assembler kernel takes a transform.
- Should `ChunkCache` be shared across `SampleStream` instances (process-global) or per-stream? Affects multi-dataset training.
- Output layout: `[N,T,C,Z,Y,X]` always, or do we expose layout choice (channels-last)?
- Do we need a CPU-staging fallback path for non-GDS systems, or is GDS a hard requirement?
- Mip selection: planner-input or auto-derived from requested aabb resolution?

## User decisions during walkthrough (2026-04-29)

Some of the open questions were resolved during the walkthrough. Recording them here for when we come back to this draft:

- AABB units: **level-0 index space** (mip downscales after).
- Mip selection: **auto** from requested aabb resolution; goal is to avoid aliasing and bound load cost to sample size.
- Cache scope: **process-global**. Inject as dependency
- GDS: **deferred** — host-pinned + cudaMemcpyAsync is the baseline. GDS is a future bonus, not a v1 target.
- Coalescing: **back in for v1** — driven by HTTP zarr v3 case where range-request consolidation is a real win, not a micro-optimization. Implies a `ChunkRequestPlanner` between `ChunkPlanner` and `ChunkLoader`.
- Codec scope: **zstd only** for the initial benchmark; blosc explicitly out.
- Read-config naming: the GLSL-style sampler concept needs a non-`Sampler` name (likely per-axis config; see project memory).
