# Distributed training

Damacy is single-rank-aware: each rank constructs its own `Pipeline`,
bound to its own GPU, draining its own slice of the sample list.
There is no cross-rank coordination inside the pipeline. Throughput
scales linearly with rank count when the storage layer keeps up.

## Binding each rank to its GPU

A multi-rank launch needs two lines that do two different things:

```python
torch.cuda.set_device(local_rank)              # for PyTorch
cfg = damacy.Config(..., device=local_rank)    # for damacy
```

`torch.cuda.set_device(local_rank)` tells **PyTorch** where new
tensors land. Without it, every rank's training tensors go to GPU 0.

`Config(device=local_rank)` tells **damacy** to retain that device's
primary CUDA context internally and run all pipeline work there.
Without `device=...`, damacy captures whatever `CUcontext` is current
on the calling thread — which silently lands every rank on GPU 0 if
`set_device` was forgotten.

Pass `local_rank` to both. The two settings are cross-checked at
construction: `Pipeline(cfg)` raises `damacy.InvalidArgument` if
`Config.device` and the already-current context disagree on a device,
and emits a `UserWarning` when `LOCAL_RANK` is set in the environment
but the bound device differs.

## A complete torchrun example

```python
# train.py — launch with:
#   torchrun --standalone --nproc_per_node=8 train.py
import os
import torch
import torch.distributed as dist
import damacy


def main() -> None:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    # See "Binding each rank to its GPU" above for why both lines.
    torch.cuda.set_device(local_rank)

    cfg = damacy.Config(
        batch_size=8,
        host_buffer_bytes=1 << 30,    # per-rank; see "Sizing per rank"
        device_buffer_bytes=1 << 30,  # per-rank
        dtype="bf16",
        n_io_threads=4,                # per-rank
        device=local_rank,
    )

    # Each rank consumes a strided slice of the sample list — see
    # "Sharding samples across ranks" below.
    all_samples = [
        damacy.Sample(uri=f"/data/cells/cell-{i}.zarr",
                      aabb=[(0, 64), (0, 256), (0, 256)])
        for i in range(8192)
    ]
    my_samples = all_samples[local_rank::world_size]

    with damacy.Pipeline(cfg) as p:
        p.push(my_samples)
        for batch in p.batches(len(my_samples) // cfg.batch_size):
            with batch as t:
                x = torch.from_dlpack(t)
                # forward / backward / optimizer step (DDP-wrapped model)
                ...

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

## Sharding samples across ranks

The example slices with stride:

```python
my_samples = all_samples[local_rank::world_size]
```

Strided sharding keeps each rank's consecutive batches drawing from
nearby URIs, which preserves shard-index and zarr-metadata cache
reuse between adjacent batches on the same rank. Contiguous sharding
(`all_samples[local_rank * n : (local_rank + 1) * n]`) also works —
use it when the original order encodes a curriculum you don't want
to interleave away.

Damacy itself doesn't enforce a sharding policy; it just consumes
whatever each rank pushes.

## Sizing per rank

Most `Config` knobs apply per rank. Aggregate cost on a node is
`(ranks per node) × (per-rank value)`:

| knob | per-rank cost |
|---|---|
| `host_buffer_bytes` | pinned RAM staging pool |
| `device_buffer_bytes` | device decompress scratch |
| `n_io_threads` | I/O worker threads |

Tune `n_io_threads` to your storage tier (NVMe pool, parallel
filesystem, object store). When stacking multiple ranks on one GPU
(uncommon, but valid), divide the device-side budgets so the per-GPU
total fits below `max_gpu_memory_bytes`.

## How damacy uses CUDA streams

Damacy creates four non-blocking CUDA streams internally (H2D,
compute, zstd, lz4) and overlaps wave-level work across them.
Nothing for you to configure, but a few facts are worth knowing:

- The streams are non-blocking, so damacy never serialises against
  the legacy default stream that some user code still lands on.
- `Batch.__dlpack__` follows the DLPack stream protocol:
  `torch.from_dlpack(batch)` passes its current PyTorch stream and
  damacy records an event/wait so the assembled tensor is fenced
  against the consuming kernels automatically. No manual events
  needed in the common case.
- For non-DLPack consumers, `BatchInfo.ready_stream` is the producer
  stream as an int handle — synchronize against it directly.
- Two pipelines on the same GPU each own their own four streams, so
  train + validation (for example) overlap up to GPU compute
  capacity without coordination.

Streams are not user-supplied. The DLPack hand-off (or
`BatchInfo.ready_stream`) is the integration boundary.

## Common failures

| symptom | likely cause |
|---|---|
| ~1/N throughput, training otherwise looks fine | missing `torch.cuda.set_device(local_rank)`. Damacy warns when `LOCAL_RANK` is set but its bound device disagrees. |
| `damacy.InvalidArgument` at `Pipeline(cfg)` with "Config.device=N but … current on device M" | `Config(device=N)` and `set_device(M)` were called for different N and M. Make them match. |
| `damacy.InvalidArgument` ("no CUcontext is current") | `Config.device` not set *and* no CUDA context primed yet. Either pass `device=local_rank` or do a `torch.empty(1, device="cuda")` first. |
| Pipeline construction succeeds but the first `pop` blocks forever | The push didn't reach the lookahead, or the consumer never frees a slot. The Python wrapper queues overflow on `push`; check `pipeline.pending` and that you're popping at the rate you push. |
