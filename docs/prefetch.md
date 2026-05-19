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
