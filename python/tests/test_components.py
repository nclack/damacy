from __future__ import annotations

import ctypes
import dataclasses
import gc
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import damacy
import numpy as np
import pytest


def zarr_metadata():
    return damacy.ZarrMetadata(
        reader=damacy.FileMetadataReader(concurrency=2),
        cache=damacy.MetadataCache(array_entries=16, shard_index_entries=64),
    )


def planner(metadata=None, **limits):
    return damacy.ChunkPlanner(
        metadata=metadata or zarr_metadata(),
        limits=damacy.PlanLimits(
            **(
                {"max_chunks": 128, "max_chunk_bytes": 1024, "max_shards_per_sample": 4}
                | limits
            )
        ),
    )


def executor(**limits):
    return damacy.CpuExecutor(
        reader=damacy.FileReader(workers=2, max_inflight_reads=1),
        limits=damacy.CpuLimits(
            **(
                {
                    "decode_workers": 2,
                    "max_encoded_chunk_bytes": 1024,
                    "max_decoded_chunk_bytes": 1024,
                    "max_memory_bytes": 8 << 20,
                }
                | limits
            )
        ),
    )


def pipeline(*, shape=(8, 16), samples=1, **overrides):
    return damacy.Pipeline(
        **(
            {
                "planner": planner(),
                "executor": executor(),
                "output": damacy.BatchSpec(samples=samples, shape=shape),
                "queues": damacy.QueueLimits(lookahead_samples=4),
                "pop_timeout_s": 5.0,
            }
            | overrides
        )
    )


def sample(uri, y=0, x=0, shape=(8, 16)):
    return damacy.Sample(uri=uri, aabb=[(y, y + shape[0]), (x, x + shape[1])])


def raw_array(path: Path, values, *, codec=None, payload=None, fill=0):
    values = np.asarray(values)
    codecs = [{"name": "bytes", "configuration": {"endian": "little"}}]
    if codec is not None:
        codecs.append(codec)
    path.mkdir()
    (path / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "array",
                "shape": list(values.shape),
                "data_type": values.dtype.name,
                "fill_value": fill,
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": list(values.shape)},
                },
                "chunk_key_encoding": {
                    "name": "default",
                    "configuration": {"separator": "/"},
                },
                "codecs": codecs,
            }
        )
    )
    chunk = path / "c" / "0" / "0"
    chunk.parent.mkdir(parents=True)
    chunk.write_bytes(values.tobytes() if payload is None else payload)
    return str(path)


def test_cpu_numpy_and_reader_backpressure(tiny_zarr):
    expected = np.arange(128, dtype=np.float32).reshape(8, 16)[1:7, 3:12]
    with pipeline(shape=(6, 9), samples=2) as p:
        p.push(sample(tiny_zarr, 1, 3, (6, 9)) for _ in range(12))
        for i in range(6):
            with p.pop() as batch:
                assert batch.info.batch_id == i
                assert batch.info.device_type == damacy.DeviceType.CPU
                assert batch.info.device_id == 0
                assert batch.info.ready_stream == 0
                assert batch.__dlpack_device__() == (1, 0)
                view = np.from_dlpack(batch)
                assert view.ctypes.data == batch.info.data
                assert view.dtype == np.float32
                np.testing.assert_array_equal(view, np.stack([expected, expected]))
                del view
        stats = p.stats()
        assert stats.gpu_bytes_committed == 0
        assert 0 < stats.host_bytes_committed <= 8 << 20
        assert stats.chunks_dispatched < stats.chunks_planned
        assert p.device == -1


def test_retained_numpy_views_and_shutdown(tiny_zarr):
    p = pipeline(shape=(2, 3), pop_timeout_s=0.05)
    p.push(
        [
            sample(tiny_zarr, 0, 0, (2, 3)),
            sample(tiny_zarr, 2, 4, (2, 3)),
            sample(tiny_zarr, 6, 10, (2, 3)),
        ]
    )
    first, second = p.pop(), p.pop()
    first_view = np.from_dlpack(first)
    expected = first_view.copy()
    first.release()
    with pytest.raises(damacy.PoolStarved):
        p.pop()
    np.testing.assert_array_equal(first_view, expected)
    second.release()
    third = p.pop()
    third_view = np.from_dlpack(third)
    third_expected = np.arange(128, dtype=np.float32).reshape(8, 16)[6:8, 10:13][None]
    third.release()
    p.close()
    del first, second, third, p
    gc.collect()
    np.testing.assert_array_equal(first_view, expected)
    np.testing.assert_array_equal(third_view, third_expected)


def test_close_wakes_timed_out_pop(tiny_zarr):
    p = pipeline(samples=2, pop_timeout_s=0.02)
    p.push([sample(tiny_zarr)])
    with pytest.raises(damacy.PoolStarved):
        p.pop()
    worker = p._pop_thread
    start = time.monotonic()
    p.close()
    assert time.monotonic() - start < 2
    assert worker is not None and not worker.is_alive()
    p.close()
    with pytest.raises(damacy.ShutdownError):
        p.pop()


def test_components_exclusive_and_reusable(tiny_zarr):
    components = {"planner": planner(), "executor": executor()}
    retained = []
    for _ in range(2):
        with pipeline(**components) as p:
            with pytest.raises(damacy.InvalidArgument):
                pipeline(**components)
            p.push([sample(tiny_zarr)])
            with p.pop() as batch:
                retained.append(np.from_dlpack(batch))
    for view in retained:
        np.testing.assert_array_equal(
            view, np.arange(128, dtype=np.float32).reshape(1, 8, 16)
        )


def test_planners_share_metadata(tiny_zarr):
    metadata = zarr_metadata()
    pipelines = [pipeline(planner=planner(metadata)) for _ in range(2)]
    try:
        for p in pipelines:
            p.push([sample(tiny_zarr)])
        for p in pipelines:
            with p.pop() as batch:
                np.testing.assert_array_equal(
                    np.from_dlpack(batch),
                    np.arange(128, dtype=np.float32).reshape(1, 8, 16),
                )
    finally:
        for p in pipelines:
            p.close()


def test_mixed_source_types(tmp_path):
    a = np.arange(16, dtype=np.uint16).reshape(4, 4)
    b = (np.arange(16, dtype=np.float32) / 8 - 2).reshape(4, 4)
    uris = [raw_array(tmp_path / "a", a), raw_array(tmp_path / "b", b)]
    with pipeline(shape=(4, 4), samples=2) as p:
        p.push(sample(uri, shape=(4, 4)) for uri in uris)
        with p.pop() as batch:
            np.testing.assert_array_equal(np.from_dlpack(batch), np.stack([a, b]))


@pytest.mark.parametrize(
    "codec,payload",
    [
        (None, b"short"),
        (
            {"name": "zstd", "configuration": {"level": 3, "checksum": False}},
            b"not a zstd frame",
        ),
        (
            {
                "name": "blosc",
                "configuration": {
                    "cname": "zstd",
                    "clevel": 3,
                    "shuffle": "shuffle",
                    "typesize": 4,
                    "blocksize": 0,
                },
            },
            bytes(16),
        ),
    ],
)
def test_corrupt_chunks_fail_terminally(tmp_path, codec, payload):
    uri = raw_array(
        tmp_path / "bad",
        np.arange(16, dtype=np.float32).reshape(4, 4),
        codec=codec,
        payload=payload,
    )
    with pipeline(shape=(4, 4)) as p:
        p.push([sample(uri, shape=(4, 4))])
        for _ in range(2):
            with pytest.raises(damacy.DecodeError):
                p.pop()


def test_memory_and_plan_limits(tiny_zarr):
    with pytest.raises(damacy.BudgetExceeded):
        pipeline(executor=executor(max_memory_bytes=4096))
    with pipeline(planner=planner(max_plan_bytes=1)) as p:
        p.push([sample(tiny_zarr)])
        with pytest.raises(damacy.BudgetExceeded):
            p.pop()
    with pipeline(executor=executor(max_decoded_chunk_bytes=8)) as p:
        p.push([sample(tiny_zarr)])
        with pytest.raises(damacy.BudgetExceeded):
            p.pop()


def test_cpu_dlpack_parameters(tiny_zarr):
    with pipeline() as p:
        p.push([sample(tiny_zarr)])
        with p.pop() as batch:
            with pytest.raises(ValueError, match="stream=None"):
                batch.__dlpack__(stream=1)
            with pytest.raises(BufferError, match="device"):
                batch.__dlpack__(dl_device=(2, 0))
            with pytest.raises(BufferError, match="copy=True"):
                batch.__dlpack__(copy=True)
            cap = batch.__dlpack__(dl_device=(1, 0), max_version=(1, 0))
            del cap


def test_invalid_limits():
    for make in [
        lambda: damacy.BatchSpec(0, (4,)),
        lambda: damacy.BatchSpec(1, (0,)),
        lambda: damacy.QueueLimits(0),
        lambda: damacy.QueueLimits(4, 0),
        lambda: damacy.CpuLimits(0),
        lambda: damacy.PlanLimits(max_chunks=0),
        lambda: damacy.CpuLimits(1 << 20, decode_workers=-1),
    ]:
        with pytest.raises(ValueError):
            make()
    with pytest.raises(TypeError):
        damacy.Pipeline(planner=planner())
    assert dataclasses.replace(damacy.BatchSpec(1, (4,)), samples=2).samples == 2


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_cpu_torch_consumer_retains_tensor(tiny_zarr, dtype):
    torch = pytest.importorskip("torch")
    with pipeline(output=damacy.BatchSpec(1, (8, 16), dtype)) as p:
        p.push([sample(tiny_zarr)])
        with p.pop() as batch:
            tensor = torch.from_dlpack(batch)
            assert tensor.device.type == "cpu"
            assert tensor.dtype == (torch.float32 if dtype == "f32" else torch.bfloat16)
    del batch
    expected = torch.arange(128, dtype=torch.float32).reshape(1, 8, 16)
    assert torch.equal(tensor.float(), expected)


@pytest.mark.usefixtures("cuda_ctx")
@pytest.mark.parametrize("read_capacity", [1, 16])
def test_cuda_composition_matches_cpu(tiny_zarr, read_capacity):
    cuda = damacy.CudaExecutor(
        reader=damacy.FileReader(workers=2, max_inflight_reads=read_capacity),
        device=0,
        limits=damacy.CudaLimits(
            max_gpu_memory_bytes=1 << 30,
            max_chunk_bytes=1024,
            max_chunks_per_wave=16,
            max_substreams_per_chunk=16,
        ),
    )
    driver = ctypes.CDLL("libcuda.so.1")
    copy = driver.cuMemcpyDtoH_v2
    copy.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
    copy.restype = ctypes.c_int
    with pipeline(shape=(6, 9), executor=cuda) as p:
        p.push([sample(tiny_zarr, 1, 3, (6, 9))])
        batch = p.pop()
        assert batch.__dlpack_device__() == (2, 0)
        result = np.empty(batch.info.shape, dtype=np.float32)
        assert copy(result.ctypes.data, batch.info.data, result.nbytes) == 0
    assert copy(result.ctypes.data, batch.info.data, result.nbytes) == 0
    capsule = batch.__dlpack__(stream=None)
    batch.release()
    del capsule
    expected = np.arange(128, dtype=np.float32).reshape(8, 16)[1:7, 3:12]
    np.testing.assert_array_equal(result, expected[None])
    with pipeline(shape=(6, 9)) as p:
        p.push([sample(tiny_zarr, 1, 3, (6, 9))])
        with p.pop() as batch:
            np.testing.assert_array_equal(result, np.from_dlpack(batch))


@pytest.mark.parametrize("shuffle", ["noshuffle", "shuffle", "bitshuffle"])
def test_blosc_filters(tmp_path, write_zarr_script, shuffle):
    uri = tmp_path / "array"
    subprocess.run(
        [
            "uv",
            "run",
            "--script",
            str(write_zarr_script),
            "--out",
            str(uri),
            "--shape",
            "8,16",
            "--inner",
            "4,8",
            "--shard",
            "8,16",
            "--dtype",
            "float32",
            "--codec",
            "blosc-zstd",
            "--shuffle",
            shuffle,
        ],
        check=True,
        capture_output=True,
    )
    with pipeline(shape=(6, 9)) as p:
        p.push([sample(str(uri), 1, 3, (6, 9))])
        with p.pop() as batch:
            expected = np.arange(128, dtype=np.float32).reshape(8, 16)[1:7, 3:12]
            np.testing.assert_array_equal(np.from_dlpack(batch), expected[None])


def test_parallel_construction_claims_components_once():
    components = {"planner": planner(), "executor": executor()}
    barrier = Barrier(2)

    def create():
        barrier.wait()
        try:
            return pipeline(**components)
        except damacy.InvalidArgument:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(create) for _ in range(2)]
        results = [future.result(timeout=5) for future in futures]
    active = [result for result in results if result is not None]
    try:
        assert len(active) == 1
    finally:
        for result in active:
            result.close()
