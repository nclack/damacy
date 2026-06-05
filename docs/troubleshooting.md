# Troubleshooting

Common errors and what to check first. For multi-GPU specifics,
see [Distributed → Common failures](distributed.md#common-failures).

## `InvalidArgument: no CUcontext is current`

`Config.device` was not set and no CUDA context has been primed on
the calling thread yet. Two fixes:

- Pass `device=0` (or `device=local_rank` under torchrun) to
  `Config`. damacy will retain that device's primary context
  internally.
- Or prime a context implicitly before constructing the pipeline:
  `torch.empty(1, device="cuda")` is enough.

## `InvalidArgument` at `Pipeline(cfg)` from metadata I/O setup

The Linux metadata path uses io_uring for zarr metadata, shard-index, and
chunk-layout reads. At construction damacy requires kernel support for
`IORING_OP_STATX`, `IORING_OP_OPENAT2`, `IORING_OP_READ`, and
`IORING_OP_CLOSE`. If ring creation or the operation probe fails,
`Pipeline(cfg)` raises `InvalidArgument` rather than falling back to a legacy
thread pool. Check the native log for the exact io_uring failure.

On supported kernels, an unusually high `metadata_io_concurrency` can also
stress process file-descriptor limits because each in-flight metadata read can
hold an open fd. The default is 32; for much deeper settings, check
`ulimit -n` and remember to multiply by ranks per node.

## `BudgetExceeded` at `Pipeline(cfg)`

`max_gpu_memory_bytes` is too small for the requested batch
geometry. The log message includes a byte-level breakdown of what
damacy tried to reserve. See [GPU memory budget → When the budget
refuses](budget.md#when-the-budget-refuses) for the full fix list;
the usual answer is to raise the cap.

## `BudgetExceeded` mid-stream

A chunk's actual uncompressed size exceeds
`Config.max_chunk_uncompressed_bytes` (default 512 KiB). Raise
that cap to fit the dataset, and raise `max_gpu_memory_bytes`
along with it if needed. See
[GPU memory budget](budget.md#when-the-budget-refuses).

## `NotFound` or `DtypeMismatch` from `pop()` (not `push()`)

`push()` only validates what's locally checkable — sample shape and
rank against `Config.sample_shape`. Errors that depend on store
contents — missing URIs, unsupported source dtypes, per-array rank
mismatch — surface from `pop()`, since damacy fetches the zarr
metadata asynchronously after `push()` returns.

```python
with Pipeline(cfg) as d:
    d.push([Sample(uri="missing", aabb=[(0, 8), (0, 16)])])  # returns fine
    try:
        d.pop()                                              # raises NotFound
    except NotFound:
        ...
```

Once any pop-side error fires, the pipeline is terminal — subsequent
`pop()` calls re-raise the same status. Build a fresh `Pipeline` to
recover.

## `PoolStarved` from `pop()`

The pool was empty for longer than `Config.pop_timeout_s`
(default 30 s). The usual cause is holding references to tensors
from previous batches — for example, stashing them in a list —
which prevents damacy from reusing the underlying slot. Drop the
references before the next `pop()`, or `.clone()` if you
genuinely need to keep them.

## `pop()` blocks forever

The push didn't reach the lookahead, or the consumer never frees
a slot:

- Read `pipeline.stats().batches_emitted` — if it stays at 0,
  pushed samples are not draining into the native lookahead.
- Make sure each `pop()` is inside a `with` block so the batch
  releases its slot before the next `pop()`.
- Verify you're popping at the rate you push (or push lazily via
  a generator so backpressure is automatic).
