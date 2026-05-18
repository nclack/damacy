# GPU memory budget

`Config.max_gpu_memory_bytes` is the single knob that tells damacy
how much GPU memory it may consume. It is a **hard cap** — damacy
refuses to allocate past it, and budget-sizing failures surface
at `Pipeline(cfg)` construction, before any streaming has started,
with a logged breakdown of what damacy tried to reserve.

**Quick answer:** 1 GiB (`1 << 30`) is a reasonable default for a
single-GPU setup. If you are sharing the GPU with a training
model, subtract the model's resident footprint (weights, optimizer
state, activations) from the device's total memory, leave a few
hundred MiB of slack for cuDNN/cuBLAS workspaces, and give damacy
the rest. damacy will use whatever you allocate to it; it will not
ask for more.

The rest of this page is for users who want to size more
precisely: a recipe for the floor, a tour of where the bytes go,
and the error paths.

## Sizing the budget

Compute the *floor* — the smallest budget that admits the
configuration — then round up generously:

```text
pool_reservation  = 2 × batch_size × prod(sample_shape) × dtype_bytes
per_wave_floor    ≈ max_chunk_uncompressed_bytes × 2   # compressed + decoded, one chunk per wave
budget_floor      ≈ pool_reservation + 2 × per_wave_floor + scratch_slack
```

The `2 ×` in front of `pool_reservation` is double-buffering on
the output side. The `2 ×` in front of `per_wave_floor` is the
two GPU waves (a fixed count, not a tunable). `scratch_slack`
covers decoder workspace and per-wave metadata — a few tens of
MiB for typical workloads, more if your chunks decompose into
many sub-streams.

`per_wave_floor` is what damacy needs to hold *one* chunk per wave
at the configured chunk cap. Any budget headroom above that floor
lets damacy size each wave to hold many chunks at once — fewer
wave turnovers per batch, better overlap.

A worked example for `batch_size=8`, `bf16` (2 bytes/element),
`sample_shape=(64, 256, 256)`, `max_chunk_uncompressed_bytes=4 MiB`:

```text
pool_reservation  = 2 × 8 × (64 × 256 × 256) × 2  ≈ 128 MiB
per_wave_floor    ≈ 4 MiB × 2                     =   8 MiB
budget_floor      ≈ 128 + 2 × 8 + slack           ≈ 200 MiB
```

Round up to the next sensible boundary — 512 MiB or 1 GiB. The
headroom does not sit idle: damacy spends it on bigger per-wave
buffers (in `max_chunk_uncompressed_bytes`-sized steps), which
shrinks wave turnover during streaming.

## How damacy uses the GPU

If you want to understand *why* the recipe looks like that, this
section walks through where damacy puts the bytes.

damacy moves data through four stages:

1. **Read** compressed chunks from a zarr store (host-side I/O).
2. **Decode** them on the GPU (zstd or blosc1-zstd, via nvcomp).
3. **Assemble** the decoded chunks into the typed samples you
   asked for, in your requested dtype.
4. **Publish** finished samples into a small pool of GPU-resident
   batches that you `pop` from.

Stages run **concurrently**. While you train on batch *N*, damacy
is already decoding chunks for batch *N+1*. That overlap is where
the throughput comes from, and it is also why GPU memory adds up
faster than you might expect.

### The unit of overlap: a wave

damacy does not stream chunks one at a time. It groups chunk work
into **waves** and keeps two waves resident on the GPU at all
times (producer wave decoding while the consumer wave assembles),
so the read, decode, assemble, and copy stages overlap.

The GPU-side wave count is fixed at two — it is not a tuning knob.
What *does* scale with the budget is **how big each wave's buffers
are**: the maximum compressed and decompressed chunk bytes the
wave can hold at once, which in turn caps how much I/O can be
issued in a single planning pass.

On the **host** side, pinned-buffer depth is tunable via
`Config.host_buffer_waves` (default 2, minimum 2). Increasing it
lets I/O for upcoming waves stay in flight while the GPU waves
turn over. Host pinned memory does *not* count against the GPU
budget.

### Where the budget is spent

Four buckets share `max_gpu_memory_bytes`:

| Bucket                  | What it holds                                              | Scales with                                              |
| ----------------------- | ---------------------------------------------------------- | -------------------------------------------------------- |
| Output batch pool       | The batches you `pop`, double-buffered                     | `batch_size`, `sample_shape`, dtype                      |
| Wave-resident buffers   | Compressed + decoded chunk bytes for the two in-flight waves | budget headroom, in `max_chunk_uncompressed_bytes` steps |
| Decoder scratch         | nvcomp's working memory                                    | Peak sub-stream count in the dataset                     |
| Per-wave metadata       | Pointer/size arrays for the decoder                        | Peak sub-stream count                                    |

The first two are the large ones; the last two are small but
*depend on the data*. damacy cannot know the sub-stream count of
a chunk until it inspects the chunk's header, so damacy picks
per-wave geometry such that even after adaptive growth to the
structural ceiling, the total fits inside the cap. Grows commit
additional bytes against `gpu_bytes_committed` as they happen,
but the geometry choice guarantees they cannot trip the cap.

## Inspecting actual usage

[`Stats.gpu_bytes_committed`](api.md#damacy.Stats) reports the
total GPU memory damacy currently holds:

```python
print(d.stats().gpu_bytes_committed)
```

It climbs as adaptive structures grow on first sight of
unusually-shaped chunks, then plateaus. It will never exceed
`max_gpu_memory_bytes`. If it sits well below your cap, you have
headroom to give back to your model.

## Two related knobs

`max_gpu_memory_bytes` is the primary cap, but two related limits
can also matter:

- **`max_chunk_uncompressed_bytes`** — rejects any individual
  chunk whose uncompressed size exceeds this value. Defaults to
  512 KiB. Raise it if your dataset uses larger chunks; otherwise
  `Pipeline(cfg)` will reject the configuration. This is a
  per-chunk sanity cap, separate from the overall budget.

- **`host_buffer_waves`** — pinned-host slab count, must be ≥ 2.
  Raising it lets host-side I/O for upcoming waves run while the
  GPU waves turn over, which can help when storage latency is the
  bottleneck. Host pinned memory does not count against
  `max_gpu_memory_bytes`.

## When the budget refuses

**`damacy.BudgetExceeded`** is the error you see when a
configured cap is too small. The common case is
`max_gpu_memory_bytes`: damacy raises it from `Pipeline(cfg)`
with a logged breakdown of what it tried to reserve, before any
streaming has started. The fix is one of:

1. Raise `max_gpu_memory_bytes`. This is the usual answer.
2. Reduce `batch_size` or `sample_shape`. The pool reservation
   scales linearly with both, and it is the part of the budget
   you control most directly.
3. Reduce `max_chunk_uncompressed_bytes` if your dataset happens
   to allow it. Chunk size sets the per-wave floor.

The less common case is `max_chunk_uncompressed_bytes`: if a
chunk's actual uncompressed size exceeds that cap, the planner
raises `BudgetExceeded` mid-stream — from the `pop()` that
touches the chunk. That is a per-chunk configuration failure,
not a budget-sizing failure; the fix is to raise
`max_chunk_uncompressed_bytes` to fit the dataset (and
`max_gpu_memory_bytes` along with it, if needed).

## What is *not* a budget error

`damacy.PoolStarved` looks superficially similar but has a
different cause: the consumer (your training loop) asked for a
batch and the pool was empty within `pop_timeout_s`. The usual
reason is your loop holding on to tensors from previous batches
(for example, by stashing them in a list), which prevents damacy
from reusing the underlying memory. Drop those references before
the next `pop()`, or call `.clone()` if you genuinely need to
keep them.

Tightening or loosening `max_gpu_memory_bytes` does not change
this behaviour.

## Summary

- One number caps every GPU-resident byte damacy uses.
- The cap is enforced at `Pipeline(cfg)`: the resolver picks
  per-wave geometry such that even after adaptive growth the
  total stays inside the cap.
- Most of the budget goes to the output batch pool and the two
  in-flight wave buffers.
- The wave *count* (2) is fixed. Per-wave buffer size scales
  with budget headroom, in `max_chunk_uncompressed_bytes` steps.
- Set `damacy.set_log_level("info")` to see the resolved
  geometry; read `Stats.gpu_bytes_committed` for runtime use.
- `BudgetExceeded` means raise the cap (or shrink the request).
  `PoolStarved` is a different problem: a consumer holding
  references.
