# Async prefetch

The [quick start](index.md#quick-start) shows the synchronous read
pattern:

```python
with batch as t:
    x = torch.from_dlpack(t)
    ...  # train step
```

`x` is a zero-copy view onto damacy's slot, and the `with` block
releases the slot at scope exit by host-syncing on damacy's
producer stream. That's fine when fwd/bwd happens inside the
block.

Training loops that prefetch the next batch on a background thread
while the main thread runs fwd/bwd need to skip that host sync.
Both patterns below do — they hand a CUDA event (or stream) to
`Batch.release(event=...)`, and damacy waits on it before reusing
the slot.

## Deferred release (preferred)

No D2D copy. The batch handle is held for the duration of the
training step; release happens after fwd/bwd has been enqueued on
the current stream:

```python
def prefetch(p):
    batch = p.pop()
    tensor = torch.from_dlpack(batch)  # zero-copy view onto damacy's slot
    return tensor, batch

# Main training loop:
tensor, batch = prefetch_future.result()
... # forward / backward on tensor
batch.release(event=torch.cuda.current_stream())  # no host sync
```

`event=torch.cuda.current_stream()` records a CUevent on the
caller's stream and tells damacy to wait on it before re-assembling
into the slot's buffer. The host returns immediately.

`torch.cuda.current_stream()` returns the per-thread currently-active
stream on the current device. By default that's the device's default
stream; inside a `with torch.cuda.stream(s):` block it returns `s`.
Call it outside any such block (as above) to capture the stream the
training step is enqueued on.

Reach for the dedicated-copy-stream pattern below instead if you
need to free damacy's slot before your training step finishes — for
example if you're chaining many small steps per batch and want
maximum overlap on slot reuse, or if the consumer can't safely hold
the slot for the whole step.

## Dedicated copy stream

Use a side stream for the D2D copy so the main stream's fwd/bwd
overlap with it, and release damacy's slot as soon as the copy is
enqueued:

```python
copy_stream = torch.cuda.Stream()

def prefetch(p):
    batch = p.pop()
    view = torch.from_dlpack(batch)
    # Allocate on the DEFAULT stream, not copy_stream — see below.
    tensor = torch.empty_like(view)
    with torch.cuda.stream(copy_stream):
        tensor.copy_(view)
    batch.release(event=copy_stream)  # slot reuse waits on copy_stream
    copy_stream.synchronize()
    return tensor
```

`copy_stream.synchronize()` blocks the prefetch thread until the
D2D copy is complete; the main thread's fwd/bwd runs in parallel
the whole time, so the host-side stall doesn't cost throughput at
prefetch depth 1. With deeper prefetch (multiple in-flight
futures), drop the `synchronize()` and have the consumer call
`torch.cuda.current_stream().wait_stream(copy_stream)` before
reading `tensor` — that pushes the wait stream-side and lets the
prefetch thread start the next pop immediately.

### Allocate on the default stream, copy on the side stream

PyTorch's caching allocator pools per stream. Allocating *inside*
`with torch.cuda.stream(copy_stream)` (with `.clone()`, or with
`torch.empty_like` placed inside the block) carves out a separate
per-stream pool that fwd/bwd can't reuse — reserved memory grows
without bound, and the training step slows down under memory
pressure.

The fix is to allocate outside the stream block (so the allocator
uses the default pool) and only the `.copy_` call inside it. This
is the single most common foot-gun with the dedicated-stream
pattern.

## `release(event=...)` accepts

`Batch.release` records or accepts a CUevent and queues a
`cuStreamWaitEvent` against damacy's internal post stream before
the slot is reused. Accepted forms:

- `None` (default) — immediate release, host-syncs the producer
  stream.
- `int` — raw CUevent handle.
- `torch.cuda.Event` — its `.cuda_event` is read.
- `torch.cuda.Stream` — `record_event()` is called on it; the
  recorded event captures the stream's then-current position.
- `cupy.cuda.Event` / `cupy.cuda.Stream` — analogous.

JAX users: JAX does not expose CUevent handles through its public
API. Drop to CUDA via DLPack and record your own event with
`cuda-python` or `cupy`.

## Why this matters

damacy's output pool is small (double-buffered by default), so
slot reuse is on the critical path of throughput. The default
`with batch as t:` block trades a host sync for simplicity — that
sync is fine for synchronous reads but stalls a prefetched loop on
every iteration. Deferred release replaces the host sync with a
stream-side wait that costs nothing on the host.

## Async Metadata I/O

The metadata prefetcher has a separate concern from batch-slot release:
hiding latency while resolving zarr metadata, shard indexes, and chunk
layouts. `prefetch_cache` has an async completion boundary: a cache miss
creates a pending slot, hands an opaque completion token to an async
fetcher, and later completes READY or ERROR without the prefetcher knowing
how the work ran.

The production metadata fetchers use a narrow async metadata-store
boundary. Array metadata submits an owned whole-file read for `zarr.json`;
shard indexes submit async stat plus an optional footer read; chunk layout
submits the 16-byte blosc header probe. The prefetcher observes only cache
state and gates:

```text
prefetcher requests metadata key
  -> prefetch cache creates async fetch state
    -> metadata store submits stat/read/open work
      -> completion pump advances cache entry
        -> prefetcher observes ready/error on the next advance
```

The metadata store is an `io_uring` driver on Linux. It submits small
metadata stat/open/read/close work without blocking the prefetcher and
completes cache entries from I/O completion state. `metadata_io_concurrency`
sets the metadata request-concurrency budget; the ring allocates enough
entries internally for multi-step requests and driver wakeups. At startup the
driver requires kernel support for `IORING_OP_STATX`, `IORING_OP_OPENAT2`,
`IORING_OP_READ`, and `IORING_OP_CLOSE`; there is no thread-pool fallback in
the current build.

The default metadata concurrency is 32. Treat much deeper values as storage
tuning: they are useful when metadata operations have real latency, but each
active read can hold an open file descriptor and allocate its destination
buffer. On multi-rank jobs, multiply the setting by ranks per node before
comparing it with `ulimit -n`.
