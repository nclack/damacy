from __future__ import annotations

import ctypes
import dataclasses
import json
import subprocess
from pathlib import Path

import damacy
import numpy as np
import pytest


def planner(**overrides):
    return damacy.ChunkPlanner(
        metadata=damacy.ZarrMetadata(
            reader=damacy.FileMetadataReader(concurrency=2),
            cache=damacy.MetadataCache(array_entries=16, shard_index_entries=64),
        ),
        limits=damacy.PlanLimits(
            **(
                {
                    "max_chunks": 128,
                    "max_chunk_bytes": 1024,
                    "max_shards_per_sample": 4,
                }
                | overrides
            )
        ),
    )


def cpu_executor():
    return damacy.CpuExecutor(
        reader=damacy.FileReader(workers=2, max_inflight_reads=1),
        limits=damacy.CpuLimits(
            max_memory_bytes=8 << 20,
            decode_workers=2,
            max_encoded_chunk_bytes=1024,
            max_decoded_chunk_bytes=1024,
        ),
    )


def pipeline(*, shape, samples=1, **overrides):
    return damacy.Pipeline(
        **(
            {
                "planner": planner(),
                "executor": cpu_executor(),
                "output": damacy.BatchSpec(samples, shape),
                "queues": damacy.QueueLimits(lookahead_samples=4),
                "pop_timeout_s": 5.0,
            }
            | overrides
        )
    )


@pytest.fixture(params=["cpu", "cuda"])
def indexed_executor(request):
    if request.param == "cpu":
        return cpu_executor()
    request.getfixturevalue("cuda_ctx")
    return damacy.CudaExecutor(
        reader=damacy.FileReader(workers=2, max_inflight_reads=1),
        device=0,
        limits=damacy.CudaLimits(
            max_gpu_memory_bytes=256 << 20,
            max_chunk_bytes=1024,
            max_chunks_per_wave=2,
            max_substreams_per_chunk=8,
        ),
    )


def read_batch(batch):
    dtype = np.float32 if batch.info.dtype == damacy.Dtype.F32 else np.uint16
    result = np.empty(batch.info.shape, dtype=dtype)
    if batch.info.device_type == damacy.DeviceType.CPU:
        ctypes.memmove(result.ctypes.data, batch.info.data, result.nbytes)
    else:
        driver = ctypes.CDLL("libcuda.so.1")
        copy = driver.cuMemcpyDtoH_v2
        copy.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        copy.restype = ctypes.c_int
        assert copy(result.ctypes.data, batch.info.data, result.nbytes) == 0
    if dtype == np.uint16:
        return (result.astype(np.uint32) << 16).view(np.float32)
    return result


def round_bfloat(values):
    bits = values.astype(np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).view(np.float32)


def raw_chunks(path, values, chunks, *, fill=0, missing=()):
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
                    "configuration": {"chunk_shape": chunks},
                },
                "chunk_key_encoding": {
                    "name": "default",
                    "configuration": {"separator": "/"},
                },
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            }
        )
    )
    grid = tuple((n + c - 1) // c for n, c in zip(values.shape, chunks, strict=True))
    for coordinate in np.ndindex(grid):
        if coordinate in missing:
            continue
        slices = tuple(
            slice(i * c, min((i + 1) * c, n))
            for i, c, n in zip(coordinate, chunks, values.shape, strict=True)
        )
        block = np.full(chunks, fill, dtype=values.dtype)
        selected = values[slices]
        block[tuple(slice(0, n) for n in selected.shape)] = selected
        file = path.joinpath("c", *map(str, coordinate))
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(block.tobytes())
    return str(path)


def test_native_axes_reject_malformed_queries(tmp_path):
    values = np.arange(16, dtype=np.float32).reshape(4, 4)
    uri = raw_chunks(tmp_path / "array", values, (2, 2))
    invalid = [
        (("unknown", (1, 3)), ValueError),
        ((None, (1, 3)), TypeError),
        ((), ValueError),
        (("interval", (1, 3), "indices"), ValueError),
        (("interval", (1, 2, 3)), ValueError),
        (("interval", None), ValueError),
        (("indices", ()), ValueError),
        (("indices", None), TypeError),
        (("indices", (1, 1.5)), TypeError),
    ]
    with pipeline(shape=(2, 2)) as p:
        for axis, error in invalid:
            with pytest.raises(error):
                p._native.push([{"uri": uri, "axes": [("indices", (3, 1)), axis]}])
        with pytest.raises(ValueError, match="only 'uri' and 'axes'"):
            p._native.push(
                [{"uri": uri, "axes": [("interval", (0, 2))] * 2, "aabb": [(0, 2)] * 2}]
            )
        with pytest.raises(KeyError, match="requires 'uri' and 'axes'"):
            p._native.push(
                [{"uri": uri, "aabb": [(0, 2)] * 2, "indices": [(1, 0)] * 2}]
            )
        p.push([damacy.IndexQuery(uri, ([3, 1], slice(1, 3)))])
        with p.pop() as batch:
            np.testing.assert_array_equal(read_batch(batch)[0], values[[3, 1], 1:3])


def test_query_copies_and_normalises_inputs():
    rows = np.array([7, 2, 7], dtype=np.int32)
    q = damacy.IndexQuery("a", (rows, slice(None, 4)))
    rows[:] = 0
    assert q.indices == ((7, 2, 7), None)
    assert q.shape == (3, 4)
    assert q == damacy.IndexQuery("a", ((7, 2, 7), slice(0, 4, 1)))
    assert hash(q) == hash(damacy.IndexQuery("a", ([7, 2, 7], slice(0, 4))))


@pytest.mark.parametrize(
    "selection",
    [
        (),
        ([],),
        ([-1],),
        ([2**63 - 1],),
        (slice(0, 0),),
        (slice(2, 1),),
        (slice(None),),
        (slice(0, 4, 2),),
    ],
)
def test_invalid_query_values(selection):
    with pytest.raises(ValueError):
        damacy.IndexQuery("a", selection)


@pytest.mark.parametrize("selection", [([1.5],), (["1"],), (1,), (slice(0.5, 2),)])
def test_non_integer_indices(selection):
    with pytest.raises(TypeError):
        damacy.IndexQuery("a", selection)


@pytest.mark.parametrize(
    "dtype", ["uint8", "uint16", "int16", "uint32", "int32", "float16", "float32"]
)
@pytest.mark.parametrize("output_dtype", ["f32", "bf16"])
def test_index_order_duplicates_and_edge_chunks(
    indexed_executor, tmp_path, dtype, output_dtype
):
    values = (np.arange(55).reshape(5, 11) * 13).astype(dtype)
    uri = raw_chunks(tmp_path / "array", values, (2, 4))
    rows, cols = [4, 0, 4, 1], [10, 3, 3, 0, 7]
    expected = values[np.ix_(rows, cols)].astype(np.float32)
    if output_dtype == "bf16":
        expected = round_bfloat(expected)
    with pipeline(
        shape=(4, 5),
        executor=indexed_executor,
        planner=planner(max_shards_per_sample=9),
        output=damacy.BatchSpec(1, (4, 5), output_dtype),
    ) as p:
        p.push([damacy.IndexQuery(uri, (rows, cols))])
        with p.pop() as batch:
            np.testing.assert_array_equal(read_batch(batch), expected[None])


@pytest.mark.parametrize(
    "selection",
    [
        ([4, 0, 4], slice(2, 7)),
        (slice(1, 4), [10, 2, 10, 0, 5]),
        ([4], [10]),
        (slice(1, 4), slice(2, 7)),
    ],
)
def test_mixed_axes(indexed_executor, tmp_path, selection):
    values = np.arange(55, dtype=np.float32).reshape(5, 11)
    uri = raw_chunks(tmp_path / "array", values, (2, 4))
    query = damacy.IndexQuery(uri, selection)
    axes = [
        np.arange(n)[axis] if isinstance(axis, slice) else axis
        for n, axis in zip(values.shape, selection, strict=True)
    ]
    expected = values[np.ix_(*axes)]
    with pipeline(
        shape=query.shape,
        executor=indexed_executor,
        planner=planner(max_shards_per_sample=9),
    ) as p:
        p.push([query])
        with p.pop() as batch:
            np.testing.assert_array_equal(read_batch(batch), expected[None])


def test_sparse_queries_only_plan_touched_chunks(indexed_executor, tmp_path):
    values = np.arange(1024, dtype=np.float32).reshape(32, 32)
    uri = raw_chunks(tmp_path / "array", values, (8, 8))
    Path(uri, "c", "1", "1").write_bytes(b"unselected corrupt chunk")
    indices = [31, 0, 31]
    with pipeline(
        shape=(3, 3), executor=indexed_executor, planner=planner(max_chunks=4)
    ) as p:
        p.push([damacy.IndexQuery(uri, (indices, indices))])
        with p.pop() as batch:
            np.testing.assert_array_equal(
                read_batch(batch), values[np.ix_(indices, indices)][None]
            )
        assert p.stats().chunks_planned == 4
        assert p.stats().assemble.output_bytes == 36


def test_mixed_queries_batches_and_repeated_indices(indexed_executor, tmp_path):
    values = np.arange(16, dtype=np.float32).reshape(4, 4)
    uri = raw_chunks(tmp_path / "array", values, (2, 2))
    query = damacy.IndexQuery(uri, ([3, 0, 3, 0], [3, 3, 0, 0]))
    rect = damacy.Sample(uri, [(0, 4), (0, 4)])
    expected = np.stack([values[np.ix_([3, 0, 3, 0], [3, 3, 0, 0])], values])
    with pipeline(shape=(4, 4), samples=2, executor=indexed_executor) as p:
        p.push(q for _ in range(5) for q in (query, rect))
        for _ in range(5):
            with p.pop() as batch:
                np.testing.assert_array_equal(read_batch(batch), expected)


def test_repeats_can_exceed_source_extent(indexed_executor, tmp_path):
    values = np.array([[3.25, -7.5]], dtype=np.float32)
    uri = raw_chunks(tmp_path / "array", values, (1, 2))
    query = damacy.IndexQuery(uri, ([0] * 17, [1, 0] * 31))
    with pipeline(shape=query.shape, executor=indexed_executor) as p:
        p.push([query])
        with p.pop() as batch:
            np.testing.assert_array_equal(
                read_batch(batch), np.tile(values[:, ::-1], (17, 31))[None]
            )


def test_missing_chunks(indexed_executor, tmp_path):
    values = np.arange(16, dtype=np.float32).reshape(4, 4)
    uri = raw_chunks(
        tmp_path / "array", values, (2, 2), fill=-7, missing=((0, 1), (1, 0))
    )
    values[:2, 2:] = -7
    values[2:, :2] = -7
    indices = [3, 0, 3]
    with pipeline(shape=(3, 3), executor=indexed_executor) as p:
        p.push([damacy.IndexQuery(uri, (indices, indices))])
        with p.pop() as batch:
            np.testing.assert_array_equal(
                read_batch(batch), values[np.ix_(indices, indices)][None]
            )


@pytest.mark.parametrize(
    "codec,shuffle",
    [
        ("zstd", "noshuffle"),
        ("blosc-zstd", "noshuffle"),
        ("blosc-zstd", "shuffle"),
        ("blosc-zstd", "bitshuffle"),
    ],
)
def test_compressed_shards(
    indexed_executor, tmp_path, write_zarr_script, codec, shuffle
):
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
            "5,11",
            "--inner",
            "2,4",
            "--shard",
            "4,8",
            "--dtype",
            "float32",
            "--codec",
            codec,
            "--shuffle",
            shuffle,
        ],
        check=True,
        capture_output=True,
    )
    rows, cols = [4, 0, 3, 4], [10, 0, 3, 10]
    expected = np.arange(55, dtype=np.float32).reshape(5, 11)[np.ix_(rows, cols)]
    with pipeline(shape=(4, 4), executor=indexed_executor) as p:
        p.push([damacy.IndexQuery(str(uri), (rows, cols))])
        with p.pop() as batch:
            np.testing.assert_array_equal(read_batch(batch), expected[None])


@pytest.mark.parametrize("rank", [1, 3, 8, 9, 31])
def test_indexed_ranks(indexed_executor, tmp_path, rank):
    shape = (1,) * (rank - 1) + (7,)
    values = np.arange(7, dtype=np.float32).reshape(shape)
    uri = raw_chunks(tmp_path / "array", values, (1,) * (rank - 1) + (3,))
    selection = ([0],) * (rank - 1) + ([6, 0, 6, 2],)
    query = damacy.IndexQuery(uri, selection)
    with pipeline(shape=query.shape, executor=indexed_executor) as p:
        p.push([query])
        with p.pop() as batch:
            np.testing.assert_array_equal(read_batch(batch).ravel(), [6, 0, 6, 2])


def test_index_bounds_and_result_shape(tmp_path):
    uri = raw_chunks(
        tmp_path / "array", np.arange(4, dtype=np.float32).reshape(2, 2), (2, 2)
    )
    with pipeline(shape=(1, 1)) as p, pytest.raises(damacy.InvalidArgument):
        p.push([damacy.IndexQuery(uri, ([0, 1], [0]))])
    with pipeline(shape=(1, 1)) as p:
        p.push([damacy.IndexQuery(uri, ([2], [0]))])
        with pytest.raises(damacy.InvalidArgument):
            p.pop()
    with (
        pipeline(shape=(1, 1), planner=planner(max_plan_bytes=16)) as p,
        pytest.raises(damacy.BudgetExceeded),
    ):
        p.push([damacy.IndexQuery(uri, ([0], [0]))])


def test_large_sparse_coordinates(indexed_executor, tmp_path):
    uri = raw_chunks(tmp_path / "array", np.array([3, 7], dtype=np.float32), (1,))
    far = 1 << 40
    metadata = Path(uri, "zarr.json")
    config = json.loads(metadata.read_text())
    config["shape"] = [far + 1]
    metadata.write_text(json.dumps(config))
    Path(uri, "c", "1").rename(Path(uri, "c", str(far)))
    with pipeline(
        shape=(3,),
        executor=indexed_executor,
        planner=planner(max_chunks=2, max_shards_per_sample=2),
    ) as p:
        p.push([damacy.IndexQuery(uri, ([far, 0, far],))])
        with p.pop() as batch:
            np.testing.assert_array_equal(read_batch(batch), [[7, 3, 7]])


def test_random_cartesian_selections(indexed_executor, tmp_path):
    values = np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)
    uri = raw_chunks(tmp_path / "array", values, (3, 3, 4))
    rng = np.random.default_rng(614)
    selections = [
        tuple(rng.integers(0, n, size=4) for n in values.shape) for _ in range(12)
    ]
    with pipeline(
        shape=(4, 4, 4),
        samples=2,
        executor=indexed_executor,
        planner=planner(max_shards_per_sample=8),
    ) as p:
        p.push(damacy.IndexQuery(uri, axes) for axes in selections)
        for i in range(0, len(selections), 2):
            expected = np.stack(
                [values[np.ix_(*axes)] for axes in selections[i : i + 2]]
            )
            with p.pop() as batch:
                np.testing.assert_array_equal(read_batch(batch), expected)


def test_cuda_index_budget(indexed_executor, tmp_path):
    if not isinstance(indexed_executor, damacy.CudaExecutor):
        pytest.skip("CUDA index storage limit")
    uri = raw_chunks(
        tmp_path / "array", np.arange(4, dtype=np.float32).reshape(2, 2), (2, 2)
    )
    for limit in (0, 31, 32):
        cuda = damacy.CudaExecutor(
            reader=indexed_executor.reader,
            limits=dataclasses.replace(indexed_executor.limits, max_index_bytes=limit),
            device=0,
        )
        with pipeline(shape=(2, 2), executor=cuda) as p:
            p.push([damacy.Sample(uri, [(0, 2), (0, 2)])])
            with p.pop() as batch:
                np.testing.assert_array_equal(read_batch(batch), [[[0, 1], [2, 3]]])
            p.push([damacy.IndexQuery(uri, ([1, 0], [1, 0]))])
            if limit < 32:
                with pytest.raises(damacy.BudgetExceeded):
                    p.pop()
            else:
                with p.pop() as batch:
                    np.testing.assert_array_equal(read_batch(batch), [[[3, 2], [1, 0]]])
            assert p.stats().gpu_bytes_committed <= cuda.limits.max_gpu_memory_bytes
