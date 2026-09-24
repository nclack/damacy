# CPU pipeline and query architecture

The CPU milestone introduces a shared preparation stage and separate CPU and
CUDA executors. Both support the existing rectangular queries. A CPU build
returns results in ordinary RAM and has no CUDA dependency. Index queries,
spatial resampling, and NGFF level selection remain later features.

The public construction and lifetime contracts are in
[Pipeline composition](../docs/pipeline.md). The C factory API is
[damacy_pipeline.h](../src/damacy_pipeline.h); Python exposes the same components.
`Pipeline(Config(...))` and `damacy_create` compose the CUDA implementation for
existing callers.

## Implemented separation

| Component | Responsibility | Implementation |
| --- | --- | --- |
| Metadata reader/provider | Zarr descriptions, shard indexes, asynchronous metadata I/O, cache capacities. | `FileMetadataReader`, `ZarrMetadata`; active caches in `pipeline/zarr_planner.c`. |
| Planner | Validate rectangular requests and publish owned source/result plans. | `ChunkPlanner`, `planner/plan_builder.c`. |
| Executor | Read encoded data, prepare codecs, decode, assemble, manage output storage. | `executor/cpu_executor.c`, `executor/cuda_executor.c`. |
| Pipeline | Bound preparation, preserve batch order, retry backpressure, publish failures, coordinate shutdown. | `damacy_plan.c`, `damacy_scheduler.c`, `damacy_pop.c`. |

The application constructs the services. Samples supply concrete array URIs;
the metadata provider loads descriptions on demand. Metadata and bulk chunk
reads use separate I/O queues. `FileReader` creates its bulk I/O queue when
constructed; no dataset is read until work is submitted. Pipeline construction starts metadata preparation and
the executor.

Planner and executor operation tables are private. `start` binds workload
geometry and resources. The planner's `next` returns one complete owned plan or
`AGAIN`. Executor `submit` accepts ownership only on `OK`; `AGAIN` leaves the
same plan and batch ID with the pipeline. The executor releases accepted plans
when execution completes or stops. `take` returns ready results in batch order.
Failures stop admission and remain visible to subsequent pops until shutdown.

A bounded queue allows metadata preparation to get ahead of execution. Owned
plans release cache protection early, so queued work does not depend on live
metadata cache entries. The metadata capacity floor is now
`lookahead_samples + samples_per_batch`, multiplied by the per-sample shard cap
for the shard cache. The legacy configuration validator keeps its older floor.

The core scheduler calls interfaces without CUDA knowledge. CUDA context
binding happens on executor thread entry and exit. CUDA staging, streams,
layout probes, wave packing, device upload, and mutable cursors live under
execution. The old internal planner entry points are thin compatibility
adapters over the shared plan builder and CUDA dispatch builder; they do not
contain another chunk-enumeration implementation.

## Owned plans

[planner/plan.h](../src/planner/plan.h) contains the private representation:

| Record | Meaning |
| --- | --- |
| Source array | Concrete array identity and a copy of its source description. |
| Source chunk | Absolute chunk coordinate, exact encoded range, decoded size, missing-chunk flag. |
| Result region | Output sample and a logical copy operation with its source region. |
| Chunk use | A connection from one distinct source chunk to one result region. |
| Output specification | Batch shape and destination dtype. |

All paths and descriptors needed for execution belong to the plan. It contains
no cache handles, CUDA pointers, streams, physical output-slot identifiers,
staging offsets, or dispatch cursors. The output geometry is independent of
the source region, though rectangular copies currently require equal extents.
Sources are assumed unchanged during use; copying metadata is not a filesystem
snapshot.

The plan records and storage are a separate library from the metadata-dependent
plan builder. Executors link the records and storage without the builder.

The CUDA executor builds its own [dispatch records](../src/executor/dispatch.h)
from this plan. Alignment and read coalescing happen there, after metadata
resolution. It preserves the existing GPU decode/assembly path. CUDA chunk
layout probes are backend preparation and never enter shared planning.

The CPU executor deduplicates decoded source chunks within a batch using the
plan's chunk-use links. Before reading, it uses the shared coalescer to merge
adjacent or overlapping ranges within each shard and interleave merged reads
across shards. Each unique chunk retains its offset within a merged read; its
uses still control output assembly. CPU reads use exact encoded byte ranges,
without CUDA's page alignment. It handles raw bytes, zstd, and C-Blosc zstd
with no, byte, or bit shuffle. It copies clipped chunk intersections and casts
supported source types to `f32` or `bf16`, including fill-only chunks. The CUDA
executor currently retains its per-use decode behavior; cross-sample GPU decode
reuse can be optimized separately.

## Resources and ownership

CPU I/O workers and decode workers are configured independently. There are two
input groups and two output buffers. Each input group holds at most
`chunks_per_input_buffer` chunks and
`chunks_per_input_buffer * max_encoded_chunk_bytes` encoded bytes. The setting
is at least `decode_workers`, so every worker can get a chunk. Merged reads
contain at most `chunks_per_input_buffer` chunks, and their count respects the
reader capacity. Decoding uses a bounded per-worker output
buffer and zstd context; memory admission also reserves conservative Blosc
workspace. The memory cap
covers executor buffers, active read plans, and temporary read-planning
scratch. It excludes metadata, reader queues, prepared-plan storage, thread
stacks, allocator overhead, and total process RSS.

The result handle has its own reference count and owns a reference to its
buffer. The executor owns another buffer reference and reuses storage only
when no result/consumer holds it. DLPack deleters release C references without
using Python objects, so consumer destruction need not run under the GIL.
Views survive batch release and pipeline shutdown. Explicit CUDA devices retain
the primary context and completion stream until the last buffer is released.
Caller-owned CUDA contexts must outlive their views.

Planners and executors reject use by two active pipelines. They can be reused
after shutdown. Metadata providers and metadata readers hold only settings that
planners copy, so planners may share them. Python retains dependencies; C
borrows them until shutdown and requires reverse-order destruction. Queue and
buffer saturation report retriable backpressure, not a storage error. Closing
a pipeline stops preparation, wakes blocked pops, joins execution, and releases
queued work before its dependencies can disappear.

## Next: index queries

An index query should accept ordered index vectors along selected dimensions,
with ordinary contiguous ranges along the others. Preserve caller order and
repeated indices. Multiple indexed dimensions should use a Cartesian product,
matching MATLAB's per-dimension indexing; this differs from paired-coordinate
point queries. A stencil can generate an index vector, so no separate stencil
query type is needed.

Extend logical result operations with compact index vectors and group chunk
uses during preparation. Keep output shape independent of the source bounding
range. Bound owned index storage along with query/plan storage. CPU assembly can
start with gather operations; CUDA can select an appropriate gather kernel.
Do not create one planning record per output voxel.

## Next: spatial resampling and NGFF

A spatial query requests a fixed output tensor, such as a transformed training
crop. Its transform maps output voxel centers into physical/source space. The
query includes sampler settings: interpolation, antialiasing/filter choices,
and boundary handling. These describe the result and belong with the query,
not in executor configuration.

Resolve an NGFF image before enumerating source chunks:

1. Load axes, units, level shapes, and level coordinate transforms.
2. Combine the output-grid transform with the level transforms and inspect
   the sampling footprint, including rotation and anisotropic scales.
3. Choose an appropriate level for the requested sampling scale and filtering
   policy, with an explicit query-level override when needed.
4. Resolve the output-to-array transform for that level and expand source
   coverage by the sampler's interpolation/filter support.
5. Publish the concrete array, resolved transform, sampler, and output geometry
   in the plan. Execution must not independently choose another level.

Exact array-index queries continue to address their specified array. NGFF
scale-dependent loading applies to spatial resampling requests. The current
`ZarrMetadata` implementation does not parse an NGFF group or choose levels;
the private interfaces leave room to add that resolution stage.

Resampling needs output tiles with all contributing source chunks available.
Keep decoded data until dependent tiles finish, and give each output tile one
writer. The shared plan already separates chunks from their uses, but the
current chunk-at-a-time rectangular executor will need this additional mode.
Boundary extension and filter halos must use the selected level's coordinates;
voxel-center conventions and downsampling quality need explicit tests.

These additions should extend logical operations and query resolution rather
than mix source discovery with codec execution. A general operation graph,
persistent decoded-chunk cache, more codecs, and ragged batching are separate
work items.

## Validation

Measured results and build evidence are recorded in
[CPU pipeline validation](cpu-pipeline-validation.md). The L40 comparison
shows 5.8% lower median CUDA throughput across three pairs on shared NFS.
The small sample and uncontrolled storage traffic do not establish a code
regression; a controlled comparison is needed to attribute the difference.

The CPU milestone checks independent crop values across codecs and source
dtypes, missing fills, bfloat16 conversion, duplicate chunk use, corrupt input,
resource limits, cache eviction, FIFO order, backpressure, and shutdown with
retained views. CPU builds and imports are checked without CUDA, including the
native library dependency chain. CUDA checks exercise both composition and the
legacy adapter, including PyTorch DLPack consumption and deferred release.

Builds and CPU tests run on CPU compute nodes. CUDA tests and paired throughput
runs use one L40. The baseline is a source snapshot taken before the refactor;
benchmark inputs, seeds, output dtype, crop shape, and measured batch counts
are held constant. CPU measurements report useful output, decoded volume,
worker scaling, and peak process RAM. They are not a cold-storage benchmark or
a TensorStore comparison.
