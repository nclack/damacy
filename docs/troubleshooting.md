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

- Check `pipeline.pending` — if it stays True, pushed samples are
  not draining into the native lookahead.
- Make sure each `pop()` is inside a `with` block so the batch
  releases its slot before the next `pop()`.
- Verify you're popping at the rate you push (or push lazily via
  a generator so backpressure is automatic).
