# Multi-GPU & distributed training

Damacy is single-rank-aware: one pipeline binds to one GPU. For DDP /
FSDP / accelerate / Lightning, instantiate one pipeline per rank and
pass `device=local_rank`. There is no cross-rank coordination inside
the pipeline; throughput scales linearly with rank count provided the
storage layer keeps up.

## Device binding model

`Config.device` selects how the pipeline acquires its CUDA context.

### `device=N` (recommended for multi-GPU)

```python
damacy.Config(..., device=local_rank)
```

The pipeline retains device `N`'s primary context internally and
pushes it on the calling thread for its lifetime. The caller does not
need to call `cuCtxSetCurrent` (or `torch.cuda.set_device`) *for
damacy's sake* — though PyTorch needs `set_device(local_rank)` for its
own tensor placement.

If a CUDA context is also current on a different device when
`Pipeline(cfg)` runs, construction fails with
`damacy.InvalidArgument`. That combination is almost always a bug
(e.g. `set_device(0)` paired with `Config(device=3)`); failing fast
beats silently landing pipeline buffers on a different GPU than your
training tensors.

### `device=None` (default)

The pipeline captures whatever `CUcontext` is current on the calling
thread. PyTorch sets one up implicitly on first allocation; bare-Python
users can call `damacy._native.cuda_init_primary()` once to prime the
primary context on device 0. With no context current, construction
raises `damacy.InvalidArgument`.

This is ergonomic for single-GPU PyTorch and for tests, but dangerous
under torchrun: forgetting `set_device(local_rank)` lets every rank's
pipeline silently bind to device 0. As a guard, damacy emits a
`UserWarning` when `LOCAL_RANK` is set in the environment but the
bound device disagrees — that's almost always a missing `set_device`.

### Which to use

| setting | when |
|---|---|
| `device=local_rank` | DDP, FSDP, accelerate, Lightning, anything launched via torchrun / mpirun / srun |
| `device=None` | single-GPU scripts, notebooks, tests, anything where there is exactly one GPU in play |

## torchrun example (8 GPUs)

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

    # PyTorch needs this for its own tensor placement; pass the same
    # rank to Config(device=...) below so damacy retains *that* device's
    # primary context internally and can't end up on the wrong GPU.
    torch.cuda.set_device(local_rank)

    cfg = damacy.Config(
        batch_size=8,
        host_buffer_bytes=1 << 30,    # per-rank
        device_buffer_bytes=1 << 30,  # per-rank
        dtype="bf16",
        n_io_threads=4,
        device=local_rank,
    )

    # Shard work by rank — strided so adjacent samples stay co-located
    # within their wave on each rank.
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

`n_io_threads` is per-rank; tune to your storage tier (NVMe pool,
parallel filesystem, or object store).
