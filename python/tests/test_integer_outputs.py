from __future__ import annotations

import ctypes
import json
import math
import struct

import damacy
import numpy as np
import pytest
from damacy import _native

INTEGER_TYPES = [
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
]


@pytest.fixture(params=["cpu", "cuda"])
def output_executor(request):
    reader = damacy.FileReader(workers=2, max_inflight_reads=4)
    if request.param == "cpu":
        return damacy.CpuExecutor(
            reader=reader,
            limits=damacy.CpuLimits(
                max_memory_bytes=8 << 20,
                decode_workers=2,
                chunks_per_input_buffer=4,
                max_encoded_chunk_bytes=4096,
                max_decoded_chunk_bytes=4096,
            ),
        )
    request.getfixturevalue("cuda_ctx")
    return damacy.CudaExecutor(
        reader=reader,
        device=0,
        limits=damacy.CudaLimits(
            max_gpu_memory_bytes=64 << 20,
            max_chunk_bytes=4096,
            max_chunks_per_wave=4,
            max_substreams_per_chunk=8,
        ),
    )


def pipeline(executor, dtype, shape, *, samples=1):
    return damacy.Pipeline(
        planner=damacy.ChunkPlanner(
            metadata=damacy.ZarrMetadata(
                reader=damacy.FileMetadataReader(concurrency=2),
                cache=damacy.MetadataCache(array_entries=16, shard_index_entries=64),
            ),
            limits=damacy.PlanLimits(
                max_chunks=128, max_chunk_bytes=4096, max_shards_per_sample=4
            ),
        ),
        executor=executor,
        output=damacy.BatchSpec(samples, shape, dtype),
        queues=damacy.QueueLimits(lookahead_samples=4),
        pop_timeout_s=5.0,
    )


def crc32c(data):
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return crc ^ 0xFFFFFFFF


def blosc_compress(values, shuffle):
    compress = ctypes.CDLL(_native.__file__).blosc_compress_ctx
    compress.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    compress.restype = ctypes.c_int
    source = np.ascontiguousarray(values)
    destination = ctypes.create_string_buffer(source.nbytes + 16)
    size = compress(
        5,
        shuffle,
        source.itemsize,
        source.nbytes,
        source.ctypes.data,
        destination,
        len(destination),
        b"zstd",
        0,
        1,
    )
    assert size > 0
    result = destination.raw[:size]
    assert not result[2] & 2
    return result


def write_array(
    path, values, *, chunks=(2, 4), shards=(4, 8), fill=0, missing=(), shuffle=None
):
    path.mkdir()
    codecs = [{"name": "bytes", "configuration": {"endian": "little"}}]
    if shuffle is not None:
        codecs.append(
            {
                "name": "blosc",
                "configuration": {
                    "cname": "zstd",
                    "clevel": 5,
                    "shuffle": ("noshuffle", "shuffle", "bitshuffle")[shuffle],
                    "typesize": values.itemsize,
                    "blocksize": 0,
                },
            }
        )
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
                    "configuration": {"chunk_shape": shards},
                },
                "chunk_key_encoding": {
                    "name": "default",
                    "configuration": {"separator": "/"},
                },
                "codecs": [
                    {
                        "name": "sharding_indexed",
                        "configuration": {
                            "chunk_shape": chunks,
                            "codecs": codecs,
                            "index_codecs": [
                                {
                                    "name": "bytes",
                                    "configuration": {"endian": "little"},
                                },
                                {"name": "crc32c"},
                            ],
                            "index_location": "end",
                        },
                    }
                ],
            }
        )
    )
    grid = tuple((n + s - 1) // s for n, s in zip(values.shape, shards, strict=True))
    inner = tuple(s // c for s, c in zip(shards, chunks, strict=True))
    for shard in np.ndindex(grid):
        payload = bytearray()
        index = bytearray()
        for local in np.ndindex(inner):
            coordinate = tuple(
                s * n + c for s, n, c in zip(shard, inner, local, strict=True)
            )
            if coordinate in missing:
                index.extend(struct.pack("<QQ", 2**64 - 1, 2**64 - 1))
                continue
            selection = tuple(
                slice(c * n, (c + 1) * n)
                for c, n in zip(coordinate, chunks, strict=True)
            )
            selected = values[selection]
            block = np.full(chunks, fill, dtype=values.dtype)
            block[tuple(slice(0, n) for n in selected.shape)] = selected
            encoded = (
                block.tobytes() if shuffle is None else blosc_compress(block, shuffle)
            )
            index.extend(struct.pack("<QQ", len(payload), len(encoded)))
            payload.extend(encoded)
        payload.extend(index)
        payload.extend(struct.pack("<I", crc32c(index)))
        file = path.joinpath("c", *map(str, shard))
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(payload)
    return str(path)


def integer_values(dtype):
    limits = np.iinfo(dtype)
    candidates = [int(limits.min), int(limits.max), int(limits.max) - 1, -1, 0, 1]
    for power in (7, 8, 15, 16, 24, 31, 32, 53, 63):
        for sign in (-1, 1):
            candidates.extend(sign * 2**power + offset for offset in (-1, 0, 1))
    values = [x for x in candidates if int(limits.min) <= x <= int(limits.max)]
    return np.resize(np.asarray(values, dtype=dtype), (8, 16))


def expected_integers(values, dtype):
    limits = np.iinfo(dtype)
    result = []
    for value in values.flat:
        if np.issubdtype(values.dtype, np.floating):
            if math.isnan(value):
                integer = 0
            elif math.isinf(value):
                integer = int(limits.max) if value > 0 else int(limits.min)
            else:
                integer = int(value)
        else:
            integer = int(value)
        result.append(min(max(integer, int(limits.min)), int(limits.max)))
    return np.array(result, dtype=dtype).reshape(values.shape)


def read_batch(batch, dtype):
    if (
        batch.info.device_type == damacy.DeviceType.CPU
        and batch.info.dtype != damacy.Dtype.BF16
    ):
        result = np.from_dlpack(batch)
        assert result.dtype == np.dtype(dtype)
        return result
    result = np.empty(batch.info.shape, dtype=dtype)
    if batch.info.device_type == damacy.DeviceType.CPU:
        ctypes.memmove(result.ctypes.data, batch.info.data, result.nbytes)
    else:
        copy = ctypes.CDLL("libcuda.so.1").cuMemcpyDtoH_v2
        copy.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        copy.restype = ctypes.c_int
        assert copy(result.ctypes.data, batch.info.data, result.nbytes) == 0
    return result


@pytest.mark.parametrize("dtype", INTEGER_TYPES)
def test_dtype_names(dtype):
    short = dtype.replace("uint", "u").replace("int", "i")
    expected = damacy.Dtype[short.upper()]
    for name in (short, dtype, dtype.upper(), expected, int(expected)):
        assert damacy.BatchSpec(1, (1,), name).dtype is expected
    assert damacy.Dtype.F32 == 0
    assert damacy.Dtype.BF16 == 1


@pytest.mark.parametrize("source_dtype", INTEGER_TYPES)
@pytest.mark.parametrize("output_dtype", INTEGER_TYPES)
def test_integer_conversions(output_executor, tmp_path, source_dtype, output_dtype):
    values = integer_values(source_dtype)
    uri = write_array(tmp_path / "array", values)
    expected = expected_integers(values, output_dtype)
    with pipeline(output_executor, output_dtype, values.shape) as p:
        p.push([damacy.Sample(uri, [(0, n) for n in values.shape])])
        with p.pop() as batch:
            result = read_batch(batch, output_dtype)
            assert batch.info.dtype is damacy.Dtype.coerce(output_dtype)
            assert result.nbytes == expected.nbytes
            np.testing.assert_array_equal(result, expected[None])
            del result
        assert p.stats().assemble.output_bytes == expected.nbytes


@pytest.mark.parametrize("dtype", INTEGER_TYPES)
@pytest.mark.parametrize("indexed", [False, True])
def test_translated_crops_and_fills(output_executor, tmp_path, dtype, indexed):
    values = integer_values(dtype)
    limits = np.iinfo(dtype)
    fill = int(limits.min) + 1 if limits.min < 0 else int(limits.max) - 1
    uri = write_array(tmp_path / "array", values, fill=fill, missing=[(1, 1), (2, 2)])
    values[2:4, 4:8] = fill
    values[4:6, 8:12] = fill
    if indexed:
        rows, cols = [6, 2, 6, 4, 1], [12, 5, 9, 3, 5]
        query = damacy.IndexQuery(uri, (rows, cols))
        expected = values[np.ix_(rows, cols)]
    else:
        query = damacy.Sample(uri, [(1, 7), (3, 13)])
        expected = values[1:7, 3:13]
    with pipeline(output_executor, dtype, expected.shape, samples=2) as p:
        p.push([query, query])
        with p.pop() as batch:
            result = read_batch(batch, dtype)
            np.testing.assert_array_equal(result, np.stack([expected, expected]))
            del result
        assert p.stats().assemble.output_bytes == 2 * expected.nbytes


@pytest.mark.parametrize("source_dtype", ["float16", "float32"])
@pytest.mark.parametrize("output_dtype", INTEGER_TYPES)
def test_float_to_integer(output_executor, tmp_path, source_dtype, output_dtype):
    values = [math.nan, math.inf, -math.inf, -0.0, 0.0, 0.99, -0.99, 1.99, -1.99]
    for power in (7, 8, 15, 16, 31, 32, 63, 64):
        for sign in (-1, 1):
            boundary = sign * 2.0**power
            if abs(boundary) <= np.finfo(source_dtype).max:
                value = np.array(boundary, dtype=source_dtype)
                values.extend(
                    [
                        float(
                            np.nextafter(value, np.array(-math.inf, dtype=source_dtype))
                        ),
                        float(value),
                        float(
                            np.nextafter(value, np.array(math.inf, dtype=source_dtype))
                        ),
                    ]
                )
    source = np.resize(np.asarray(values, dtype=source_dtype), (8, 16))
    uri = write_array(tmp_path / "array", source)
    expected = expected_integers(source, output_dtype)
    with pipeline(output_executor, output_dtype, source.shape) as p:
        p.push([damacy.Sample(uri, [(0, 8), (0, 16)])])
        with p.pop() as batch:
            np.testing.assert_array_equal(
                read_batch(batch, output_dtype), expected[None]
            )


@pytest.mark.parametrize("output_dtype", INTEGER_TYPES)
@pytest.mark.parametrize("fill", [-1.9, "NaN", "Infinity", "-Infinity"])
def test_float_fill_conversion(output_executor, tmp_path, output_dtype, fill):
    values = np.full((8, 16), fill, dtype=np.float32)
    uri = write_array(
        tmp_path / "array", values, fill=fill, missing=list(np.ndindex((4, 4)))
    )
    expected = expected_integers(values, output_dtype)
    with pipeline(output_executor, output_dtype, values.shape) as p:
        p.push([damacy.Sample(uri, [(0, 8), (0, 16)])])
        with p.pop() as batch:
            np.testing.assert_array_equal(
                read_batch(batch, output_dtype), expected[None]
            )


@pytest.mark.parametrize("source_dtype", ["int8", "uint64", "int64"])
@pytest.mark.parametrize("output_dtype", ["f32", "bf16"])
def test_new_sources_float_outputs(
    output_executor, tmp_path, source_dtype, output_dtype
):
    values = integer_values(source_dtype)
    uri = write_array(tmp_path / "array", values)
    expected = values.astype(np.float32)
    if output_dtype == "bf16":
        bits = expected.view(np.uint32)
        expected = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    with pipeline(output_executor, output_dtype, values.shape) as p:
        p.push([damacy.Sample(uri, [(0, 8), (0, 16)])])
        with p.pop() as batch:
            np.testing.assert_array_equal(
                read_batch(batch, expected.dtype), expected[None]
            )


@pytest.mark.parametrize(
    "dtype", ["uint16", "int16", "uint32", "int32", "uint64", "int64"]
)
@pytest.mark.parametrize("shuffle", [1, 2])
def test_shuffled_integer_chunks(output_executor, tmp_path, dtype, shuffle):
    values = np.resize(integer_values(dtype), (16, 32))
    uri = write_array(
        tmp_path / "array", values, chunks=(8, 16), shards=(16, 32), shuffle=shuffle
    )
    with pipeline(output_executor, dtype, (13, 27)) as p:
        p.push([damacy.Sample(uri, [(1, 14), (3, 30)])])
        with p.pop() as batch:
            np.testing.assert_array_equal(
                read_batch(batch, dtype), values[None, 1:14, 3:30]
            )


@pytest.mark.parametrize("dtype", INTEGER_TYPES)
def test_torch_dtype_and_retained_values(output_executor, tmp_path, dtype):
    torch = pytest.importorskip("torch")
    values = integer_values(dtype)
    uri = write_array(tmp_path / "array", values)
    with pipeline(output_executor, dtype, values.shape) as p:
        p.push([damacy.Sample(uri, [(0, 8), (0, 16)])])
        with p.pop() as batch:
            result = torch.from_dlpack(batch)
            assert result.dtype == getattr(torch, dtype)
            assert result.element_size() == values.itemsize
            assert result.device.type == (
                "cpu" if batch.info.device_type == damacy.DeviceType.CPU else "cuda"
            )
    np.testing.assert_array_equal(result.cpu().numpy(), values[None])


@pytest.mark.parametrize("output_dtype", [*INTEGER_TYPES, "f32", "bf16"])
def test_reject_float64_sources(output_executor, tmp_path, output_dtype):
    values = np.ones((8, 16), dtype=np.float64)
    uri = write_array(tmp_path / "array", values)
    with pipeline(output_executor, output_dtype, values.shape) as p:
        p.push([damacy.Sample(uri, [(0, 8), (0, 16)])])
        with pytest.raises(damacy.DtypeMismatch):
            p.pop()
