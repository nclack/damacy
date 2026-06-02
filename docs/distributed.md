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
from collections.abc import Iterator

import damacy
import torch
import torch.distributed as dist


def crops_for_rank(rank: int, world_size: int) -> Iterator[damacy.Sample]:
    """Yield an indefinite stream of `damacy.Sample` for this rank.
    Sharding policy lives here — see "Sharding samples across ranks"
    below. The main README covers sample-construction patterns
    (random crops, tile grids, curriculums)."""
    ...


def main() -> None:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    # See "Binding each rank to its GPU" above for why both lines.
    torch.cuda.set_device(local_rank)

    cfg = damacy.Config(
        samples_per_batch=8,
        sample_shape=(64, 256, 256),
        max_gpu_memory_bytes=1 << 30,   # per-rank GPU budget
        dtype="bf16",
        n_io_threads=4,                 # bulk chunk reads, per-rank
        n_prefetch_threads=16,          # metadata dependency workers, per-rank
        n_metadata_io_threads=8,        # metadata store workers, per-rank
        device=local_rank,
    )

    model = ...        # your DDP-wrapped module
    optimizer = ...    # your optimizer
    n_steps = 10_000

    with damacy.Pipeline(cfg) as p:
        # Hand the generator off once; the pipeline pulls lazily as
        # pops free space, so memory stays bounded even though
        # crops_for_rank is unbounded.
        p.push(crops_for_rank(local_rank, world_size))

        for step in range(n_steps):
            with p.pop() as batch:
                x = torch.from_dlpack(batch)  # zero-copy, stream-fenced
                # forward / backward / optimizer step:
                # loss = model(x).mean(); loss.backward(); optimizer.step()
                ...

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

## Sharding samples across ranks

`crops_for_rank` decides which samples each rank sees. Damacy
doesn't enforce a sharding policy — whatever your generator yields
is what that rank works on. Two common patterns:

- **Strided** (`all_samples[rank::world_size]`): each rank's
  consecutive batches draw from nearby URIs, which preserves
  shard-index and zarr-metadata cache reuse between adjacent batches
  on the same rank.
- **Contiguous** (`all_samples[rank * n : (rank + 1) * n]`): use
  this when the original order encodes a curriculum you don't want
  to interleave away.

Generators that draw random crops can shard implicitly by seeding
their RNG with `(base_seed, rank, epoch)` so each rank's sequence is
deterministic and disjoint.

## Sizing per rank

Most `Config` knobs apply per rank. Aggregate cost on a node is
`(ranks per node) × (per-rank value)`:

| knob | per-rank cost |
|---|---|
| `max_gpu_memory_bytes` | GPU budget: wave buffers, decoder scratch, batch-output pool |
| `host_buffer_waves` | pinned-host slab pool, in waves |
| `n_io_threads` | bulk chunk-read worker threads |
| `n_prefetch_threads` | metadata dependency-resolution worker threads |
| `n_metadata_io_threads` | metadata store backend worker threads |
| `n_array_meta_cache`, `n_shard_index_cache`, `n_chunk_layout_cache` | LRU caps for parsed metadata |

Tune `n_io_threads` to your storage tier's bulk read behavior, and keep
`n_prefetch_threads` high enough to cover small, latency-bound metadata
dependency work. `n_metadata_io_threads` controls the metadata store's
backend worker pool. When stacking multiple ranks on one GPU (uncommon,
but valid), divide `max_gpu_memory_bytes` so the per-GPU total fits
within the device.

## NUMA placement

On multi-socket hosts, `Config.numa_strategy` controls where damacy
pins its pinned-host slabs and worker threads:

- `NumaStrategy.AUTO` (default) resolves the GPU's host-NUMA node
  from the CUDA driver and pins everything there. On single-node
  hosts the strategy is a no-op.
- `NumaStrategy.PIN_TO` plus `numa_node=N` forces pinning to node
  `N`. Use this when AUTO can't resolve a node (no sysfs topology
  available) or when you want each rank explicitly pinned to a
  specific socket.
- `NumaStrategy.DISABLED` skips all NUMA work, including the
  driver query at create time.

All three are silent no-ops if `libnuma.so.1` cannot be loaded at
runtime; damacy logs the no-op reason and falls back to whatever
the OS scheduler picks.

## CUDA streams

DLPack handles synchronization. `torch.from_dlpack(batch)` records
an event on damacy's output stream and makes the consuming PyTorch
stream wait on it, so train kernels see a fenced tensor.

Damacy's internal streams are non-blocking, so a default-stream
operation elsewhere in your code won't accidentally serialize the
pipeline. The flip side: don't assume the legacy default stream
sees damacy's work either. If you bypass DLPack and read
`BatchInfo.device_ptr` directly, do a `cuStreamWaitEvent` against
`BatchInfo.ready_stream` on your consumer stream before launching
your kernel — the implicit default-stream sync is gone.

## Common failures

Multi-rank-specific symptoms. For single-GPU errors
(`no CUcontext is current`, blocking `pop()`, `BudgetExceeded`,
`PoolStarved`), see [Troubleshooting](troubleshooting.md).

| symptom | likely cause |
|---|---|
| ~1/N throughput, training otherwise looks fine | missing `torch.cuda.set_device(local_rank)`. Damacy warns when `LOCAL_RANK` is set but its bound device disagrees. |
| `damacy.InvalidArgument` at `Pipeline(cfg)` with "Config.device=N but … current on device M" | `Config(device=N)` and `set_device(M)` were called for different N and M. Make them match. |
