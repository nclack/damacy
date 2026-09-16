from __future__ import annotations

import ctypes
import dataclasses
import gc
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import damacy
import numpy as np
import pytest


def write_image(path, *, shape=(32, 32), axes=None, scales=None, data=False):
    path.mkdir()
    rank = len(shape)
    axes = axes or [(name, "space") for name in ("z", "y", "x")[-rank:]]
    scales = scales or [
        [2**level if kind == "space" else 1 for _, kind in axes] for level in range(3)
    ]
    datasets = []
    arrays = []
    for level, scale in enumerate(scales):
        level_shape = tuple(int(n / s) for n, s in zip(shape, scale, strict=True))
        values = (
            np.arange(np.prod(level_shape), dtype=np.uint16).reshape(level_shape)
            + level * 10000
        )
        arrays.append(values)
        array_path = path / str(level)
        array_path.mkdir()
        chunks = [
            min(8, n) if kind == "space" else 1
            for n, (_, kind) in zip(level_shape, axes, strict=True)
        ]
        metadata = {
            "zarr_format": 3,
            "node_type": "array",
            "shape": level_shape,
            "dimension_names": [name for name, _ in axes],
            "data_type": "uint16",
            "fill_value": 0,
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": chunks}},
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        }
        (array_path / "zarr.json").write_text(json.dumps(metadata))
        datasets.append(
            {
                "path": str(level),
                "coordinateTransformations": [
                    {"type": "scale", "scale": scale},
                    {
                        "type": "translation",
                        "translation": [(s - 1) / 2 for s in scale],
                    },
                ],
            }
        )
        if data:
            grid = tuple(
                (n + c - 1) // c for n, c in zip(level_shape, chunks, strict=True)
            )
            for coordinate in np.ndindex(grid):
                slices = tuple(
                    slice(i * c, min((i + 1) * c, n))
                    for i, c, n in zip(coordinate, chunks, level_shape, strict=True)
                )
                block = np.zeros(chunks, dtype=np.uint16)
                selected = values[slices]
                block[tuple(slice(0, n) for n in selected.shape)] = selected
                file = array_path.joinpath("c", *map(str, coordinate))
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(block.tobytes())
    metadata = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5",
                "multiscales": [
                    {
                        "axes": [{"name": name, "type": kind} for name, kind in axes],
                        "datasets": datasets,
                    }
                ],
            }
        },
    }
    (path / "zarr.json").write_text(json.dumps(metadata))
    return arrays


def edit(path, keys, value):
    metadata = json.loads(path.read_text())
    target = metadata
    for key in keys[:-1]:
        target = target[key]
    if value is None:
        del target[keys[-1]]
    else:
        target[keys[-1]] = value
    path.write_text(json.dumps(metadata))


def load(path, **kwargs):
    return damacy.NgffImage(
        path,
        reader=damacy.FileMetadataReader(concurrency=2),
        multiscale_index=0,
        **kwargs,
    )


def query(
    scale=1, offset=(0, 0), *, sampler=None, level: int | Literal["auto"] = "auto"
):
    rank = len(offset)
    return damacy.SpatialQuery(
        output_to_reference=tuple(
            (*(scale if i == j else 0 for j in range(rank)), origin)
            for i, origin in enumerate(offset)
        ),
        sampler=sampler or damacy.Sampler(filter="nearest"),
        level=level,
    )


@pytest.fixture(params=["cpu", "cuda"])
def spatial_executor(request):
    reader = damacy.FileReader(workers=2, max_inflight_reads=2)
    if request.param == "cpu":
        return damacy.CpuExecutor(
            reader=reader,
            limits=damacy.CpuLimits(
                max_memory_bytes=8 << 20,
                decode_workers=2,
                max_encoded_chunk_bytes=4096,
                max_decoded_chunk_bytes=4096,
            ),
        )
    request.getfixturevalue("cuda_ctx")
    return damacy.CudaExecutor(
        reader=reader,
        device=0,
        limits=damacy.CudaLimits(
            max_gpu_memory_bytes=256 << 20,
            max_chunk_bytes=4096,
            max_chunks_per_wave=2,
            max_substreams_per_chunk=8,
        ),
    )


def pipeline(executor, output, reader=None):
    return damacy.Pipeline(
        planner=damacy.ChunkPlanner(
            metadata=damacy.ZarrMetadata(
                reader=reader or damacy.FileMetadataReader(concurrency=2),
                cache=damacy.MetadataCache(array_entries=16, shard_index_entries=32),
            ),
            limits=damacy.PlanLimits(
                max_chunks=128, max_chunk_bytes=4096, max_shards_per_sample=4
            ),
        ),
        executor=executor,
        output=output,
        queues=damacy.QueueLimits(lookahead_samples=4),
        pop_timeout_s=5,
    )


def read_batch(batch):
    result = np.empty(batch.info.shape, dtype=np.float32)
    if batch.info.device_type == damacy.DeviceType.CPU:
        ctypes.memmove(result.ctypes.data, batch.info.data, result.nbytes)
    else:
        driver = ctypes.CDLL("libcuda.so.1")
        copy = driver.cuMemcpyDtoH_v2
        copy.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        copy.restype = ctypes.c_int
        assert copy(result.ctypes.data, batch.info.data, result.nbytes) == 0
    return result


@pytest.mark.parametrize("level", [0, 1, 2])
def test_resolved_crops_decode_at_selected_level(tmp_path, spatial_executor, level):
    root = tmp_path / "image"
    arrays = write_image(root, data=True)
    reader = damacy.FileMetadataReader(concurrency=2)
    image = damacy.NgffImage(root, reader=reader, multiscale_index=0)
    output = damacy.BatchSpec(2, (4, 4))
    shape = output.shape
    factor = 2**level
    resolved = image.resolve(query(factor, (factor * 2, factor * 3)), shape=shape)
    assert resolved.level == level
    assert not resolved.requires_resampling
    assert resolved.source_bounds_index == ((2, 6), (3, 7))
    assert resolved.output_to_source == ((1, 0, 2), (0, 1, 3))
    del image
    gc.collect()
    with pipeline(spatial_executor, output, reader) as p:
        p.push([resolved, pickle.loads(pickle.dumps(resolved))])
        with p.pop() as batch:
            expected = arrays[level][2:6, 3:7]
            np.testing.assert_array_equal(
                read_batch(batch), np.stack([expected, expected])
            )


def test_resampling_fails_before_decoding_and_pipeline_recovers(
    tmp_path, spatial_executor
):
    root = tmp_path / "image"
    arrays = write_image(root, data=True)
    output = damacy.BatchSpec(1, (4, 4))
    image = load(root)
    shape = output.shape
    rotated = image.resolve(
        damacy.SpatialQuery(
            output_to_reference=((1, -1, 12), (1, 1, 4)),
            sampler=damacy.Sampler(filter="linear"),
        ),
        shape=shape,
    )
    assert rotated.requires_resampling
    with pipeline(spatial_executor, output) as p:
        with pytest.raises(damacy.UnsupportedOperation, match="resampling required"):
            p.push([rotated])
        p.push([image.resolve(query(), shape=shape)])
        with p.pop() as batch:
            np.testing.assert_array_equal(read_batch(batch)[0], arrays[0][:4, :4])


def test_fixed_shape_is_checked_at_push(tmp_path, spatial_executor):
    root = tmp_path / "image"
    write_image(root)
    resolved = load(root).resolve(query(), shape=(3, 3))
    with (
        pipeline(spatial_executor, damacy.BatchSpec(1, (4, 4))) as p,
        pytest.raises(damacy.InvalidArgument),
    ):
        p.push([resolved])


def test_ngff_center_offsets_stay_in_adapter(tmp_path):
    root = tmp_path / "image"
    write_image(root)
    path = root / "zarr.json"
    prefix = ("attributes", "ome", "multiscales", 0)
    transforms = (
        [
            {"type": "scale", "scale": [0.5, 1]},
            {"type": "translation", "translation": [10, 20]},
        ],
        [
            {"type": "scale", "scale": [1, 2]},
            {"type": "translation", "translation": [10.25, 20.5]},
        ],
        [
            {"type": "scale", "scale": [2, 4]},
            {"type": "translation", "translation": [10, 20]},
        ],
    )
    for i, value in enumerate(transforms):
        edit(path, (*prefix, "datasets", i, "coordinateTransformations"), value)
    edit(
        path,
        (*prefix, "coordinateTransformations"),
        [
            {"type": "scale", "scale": [3, 7]},
            {"type": "translation", "translation": [900, -100]},
        ],
    )
    image = load(root)
    assert image.levels[0].origin_reference_index == (0, 0)
    assert image.levels[1].origin_reference_index == (0, 0)
    assert image.levels[2].origin_reference_index == (-1.5, -1.5)
    shape = (4, 4)
    resolved = image.resolve(query(2), shape=shape)
    assert resolved.level == 1 and not resolved.requires_resampling
    assert image.resolve(query(4), shape=shape).requires_resampling


def test_anisotropic_rotated_level_selection_and_time_channel_axes(tmp_path):
    root = tmp_path / "image"
    write_image(
        root,
        shape=(2, 3, 16, 16, 16),
        axes=[
            ("t", "time"),
            ("c", "channel"),
            ("z", "space"),
            ("y", "space"),
            ("x", "space"),
        ],
        scales=[[1, 1, 1, 1, 1], [1, 1, 1, 2, 4]],
    )
    image = load(root)
    shape = (1, 1, 2, 2, 2)
    transform = np.zeros((5, 6))
    transform[:5, :5] = np.eye(5)
    transform[:2, 5] = (1, 2)
    transform[2:5, 2:5] = np.array([[0, -1, 0], [2, 0, 0], [0, 0, 4]])
    transform[2:, 5] = (4, 4, 0)
    resolved = image.resolve(
        damacy.SpatialQuery(
            output_to_reference=transform, sampler=damacy.Sampler(filter="nearest")
        ),
        shape=shape,
    )
    assert resolved.level == 1
    assert resolved.output_to_source[2] == (0, 0, 0, -1, 0, 4)
    assert resolved.output_to_source[3] == (0, 0, 1, 0, 0, 2)
    assert resolved.source_bounds_index[:2] == ((1, 2), (2, 3))
    transform[2, 3] = -0.5
    assert (
        image.resolve(
            damacy.SpatialQuery(
                output_to_reference=transform, sampler=damacy.Sampler()
            ),
            shape=shape,
        ).level
        == 0
    )
    transform[0, 0] = 2
    with pytest.raises(damacy.InvalidArgument):
        image.resolve(
            damacy.SpatialQuery(
                output_to_reference=transform, sampler=damacy.Sampler()
            ),
            shape=shape,
        )


def test_selection_uses_smallest_spacing_not_column_lengths(tmp_path):
    root = tmp_path / "image"
    write_image(root)
    image = load(root)
    shape = (2, 2)
    q = damacy.SpatialQuery(
        output_to_reference=((3, 2, 0), (2, 3, 0)),
        sampler=damacy.Sampler(filter="nearest"),
    )
    assert image.resolve(q, shape=shape).level == 0
    forced = dataclasses.replace(q, level=1)
    assert image.resolve(forced, shape=shape).level == 1


@pytest.mark.parametrize(
    "filter,boundary,offset,source,reads",
    [
        ("nearest", "error", -0.25, (0, 4), (0, 4)),
        ("linear", "constant", -0.25, (-1, 4), (0, 4)),
        ("linear", "clamp", -0.25, (-1, 4), (0, 4)),
        ("nearest", "constant", -10, (-10, -6), (0, 0)),
        ("nearest", "clamp", -10, (-10, -6), (0, 1)),
        ("nearest", "constant", 40, (40, 44), (32, 32)),
        ("nearest", "clamp", 40, (40, 44), (31, 32)),
    ],
)
def test_source_and_read_bounds(tmp_path, filter, boundary, offset, source, reads):
    root = tmp_path / "image"
    write_image(root)
    image = load(root)
    shape = (4, 4)
    resolved = image.resolve(
        query(
            offset=(offset, 0), sampler=damacy.Sampler(filter=filter, boundary=boundary)
        ),
        shape=shape,
    )
    assert resolved.source_bounds_index[0] == source
    assert resolved.read_bounds_index[0] == reads
    assert resolved.requires_resampling


@pytest.mark.parametrize(
    "transform",
    [
        ((0, 0, 0), (0, 1, 0)),
        ((1, 2, 0), (2, 4, 0)),
        ((1, 0, -1), (0, 1, 0)),
        ((1e100, 0, 0), (0, 1, 0)),
        ((1, 0, 2**52), (0, 1, 0)),
    ],
)
def test_invalid_geometry(tmp_path, transform):
    root = tmp_path / "image"
    write_image(root)
    image = load(root)
    shape = (4, 4)
    with pytest.raises(damacy.InvalidArgument):
        image.resolve(
            damacy.SpatialQuery(
                output_to_reference=transform, sampler=damacy.Sampler()
            ),
            shape=shape,
        )


@pytest.mark.parametrize(
    "keys,value",
    [
        (("version",), "0.4"),
        (("multiscales", 0, "axes", 0, "name"), "x"),
        (("multiscales", 0, "axes", 0, "type"), "custom"),
        (("multiscales", 0, "datasets", 1, "path"), "0"),
        (("multiscales", 0, "datasets", 1, "path"), "../outside"),
        (("multiscales", 0, "datasets", 1, "path"), "/outside"),
        (("multiscales", 0, "datasets", 1, "path"), "1//array"),
        (
            ("multiscales", 0, "datasets", 1, "coordinateTransformations", 0, "scale"),
            [2],
        ),
        (
            ("multiscales", 0, "datasets", 1, "coordinateTransformations", 0, "scale"),
            [0, 2],
        ),
        (
            ("multiscales", 0, "datasets", 1, "coordinateTransformations", 0, "scale"),
            [0.5, 2],
        ),
        (
            ("multiscales", 0, "datasets", 1, "coordinateTransformations", 0, "scale"),
            [-2, 2],
        ),
        (
            ("multiscales", 0, "datasets", 1, "coordinateTransformations", 0, "type"),
            "affine",
        ),
        (("multiscales", 0, "datasets", 1, "coordinateTransformations"), []),
    ],
)
def test_rejects_unsupported_or_ambiguous_metadata(tmp_path, keys, value):
    root = tmp_path / "image"
    write_image(root)
    edit(root / "zarr.json", ("attributes", "ome", *keys), value)
    with pytest.raises((damacy.InvalidArgument, damacy.UnsupportedOperation)):
        load(root)


@pytest.mark.parametrize(
    "key,value",
    [
        ("dimension_names", ["x", "y"]),
        ("dimension_names", None),
        ("dimension_names", ["y"]),
        ("shape", [16, 16, 16]),
        ("shape", [64, 16]),
        ("data_type", "float32"),
    ],
)
def test_level_arrays_must_agree_with_ngff(tmp_path, key, value):
    root = tmp_path / "image"
    write_image(root)
    edit(root / "1" / "zarr.json", (key,), value)
    with pytest.raises(damacy.DamacyError):
        load(root)


def test_limits_errors_and_reader_reuse(tmp_path):
    root = tmp_path / "image"
    write_image(root)
    reader = damacy.FileMetadataReader(concurrency=2)
    metadata_bytes = sum(p.stat().st_size for p in root.rglob("zarr.json"))
    for limits in [
        damacy.NgffLimits(max_levels=2),
        damacy.NgffLimits(max_metadata_bytes=metadata_bytes - 1),
        damacy.NgffLimits(max_metadata_bytes=1),
    ]:
        with pytest.raises(damacy.BudgetExceeded):
            damacy.NgffImage(root, reader=reader, multiscale_index=0, limits=limits)
    image = damacy.NgffImage(
        root,
        reader=reader,
        multiscale_index=0,
        limits=damacy.NgffLimits(max_metadata_bytes=metadata_bytes),
    )
    assert image.levels[0].shape == (32, 32)
    with pytest.raises(damacy.NotFound):
        load(tmp_path / "missing")
    with pytest.raises(damacy.InvalidArgument):
        damacy.NgffImage(root, reader=reader, multiscale_index=1)


def test_multiscale_selection_and_escaped_strings(tmp_path):
    root = tmp_path / "image"
    write_image(root, axes=[("μ", "space"), ("x", "space")])
    path = root / "zarr.json"
    metadata = json.loads(path.read_text())
    entry = metadata["attributes"]["ome"]["multiscales"][0]
    metadata["attributes"]["ome"]["multiscales"].insert(0, {})
    path.write_text(
        json.dumps(metadata)
        .replace('"version"', '"ver\\u0073ion"')
        .replace('"0.5"', '"0.\\u0035"')
    )
    image = damacy.NgffImage(
        root, reader=damacy.FileMetadataReader(concurrency=2), multiscale_index=1
    )
    assert image.axes[0].name == "μ"
    assert entry["axes"][0]["name"] == image.axes[0].name
    with pytest.raises(damacy.InvalidArgument):
        load(root)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda text: text.replace(
            '"version": "0.5"', '"version": "0.5", "version": "0.4"'
        ),
        lambda text: text.replace(
            '"scale": [1, 1]', '"scale": [1, 1], "scale": [2, 2]'
        ),
        lambda text: text[:-1] + ",}",
    ],
)
def test_invalid_consumed_metadata_is_rejected(tmp_path, mutate):
    root = tmp_path / "image"
    write_image(root)
    path = root / "zarr.json"
    path.write_text(mutate(path.read_text()))
    with pytest.raises(damacy.DamacyError):
        load(root)


def test_resolution_is_independent_of_files_and_can_run_in_threads(tmp_path):
    root = tmp_path / "image"
    write_image(root)
    image = load(root)
    shape = (4, 4)
    for metadata in root.rglob("zarr.json"):
        metadata.unlink()
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(
            workers.map(lambda q: image.resolve(q, shape=shape), [query(2)] * 16)
        )
    assert all(
        r.level == 1 and r.source_bounds_index == ((0, 4), (0, 4)) for r in results
    )


def test_query_values_are_copied_and_validated():
    matrix = [[1, 0, 2], [0, 1, 3]]
    q = damacy.SpatialQuery(output_to_reference=matrix, sampler=damacy.Sampler())
    matrix[0][2] = 100
    assert q.output_to_reference[0][2] == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        q.level = 2  # type: ignore[misc]
    for invalid in [(), ((1, 0), (0, 1)), ((1, 0, float("nan")), (0, 1, 0))]:
        with pytest.raises(ValueError):
            damacy.SpatialQuery(output_to_reference=invalid, sampler=damacy.Sampler())
    with pytest.raises(ValueError):
        damacy.Sampler(boundary="clamp", constant_value=3)
    with pytest.raises(ValueError):
        damacy.NgffLimits(max_levels=0)


def test_anisotropic_crop_decodes_with_time_and_channel_axes(
    tmp_path, spatial_executor
):
    root = tmp_path / "image"
    arrays = write_image(
        root,
        shape=(2, 3, 16, 16, 16),
        axes=[
            ("t", "time"),
            ("c", "channel"),
            ("z", "space"),
            ("y", "space"),
            ("x", "space"),
        ],
        scales=[[1, 1, 1, 1, 1], [1, 1, 1, 2, 4]],
        data=True,
    )
    output = damacy.BatchSpec(1, (1, 2, 4, 4, 4))
    image = load(root)
    shape = output.shape
    transform = np.column_stack((np.diag([1, 1, 1, 2, 4]), [1, 1, 4, 4, 0]))
    resolved = image.resolve(
        damacy.SpatialQuery(output_to_reference=transform, sampler=damacy.Sampler()),
        shape=shape,
    )
    assert resolved.level == 1 and not resolved.requires_resampling
    with pipeline(spatial_executor, output) as p:
        p.push([resolved])
        with p.pop() as batch:
            np.testing.assert_array_equal(
                read_batch(batch)[0], arrays[1][1:2, 1:3, 4:8, 2:6, :4]
            )


def test_automatic_level_selection_matches_singular_values(tmp_path):
    root = tmp_path / "image"
    scales = np.array([[1, 1, 1], [1, 2, 4], [2, 4, 8]])
    write_image(root, shape=(32, 32, 32), scales=scales.tolist())
    image = load(root)
    shape = (2, 2, 2)
    rng = np.random.default_rng(532)
    for _ in range(64):
        left, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        right, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        matrix = left @ np.diag(2 ** rng.uniform(-2, 4, 3)) @ right
        expected = 0
        for level, scale in enumerate(scales):
            if np.linalg.svd(matrix / scale[:, None], compute_uv=False)[-1] >= 1:
                expected = level
        resolved = image.resolve(
            damacy.SpatialQuery(
                output_to_reference=np.column_stack((matrix, [8, 8, 8])),
                sampler=damacy.Sampler(boundary="constant"),
            ),
            shape=shape,
        )
        assert resolved.level == expected


def test_active_metadata_reader_is_not_reused(tmp_path, spatial_executor):
    root = tmp_path / "image"
    write_image(root)
    reader = damacy.FileMetadataReader(concurrency=2)
    with (
        pipeline(spatial_executor, damacy.BatchSpec(1, (4, 4)), reader),
        pytest.raises(damacy.InvalidArgument),
    ):
        damacy.NgffImage(root, reader=reader, multiscale_index=0)
    assert damacy.NgffImage(root, reader=reader, multiscale_index=0).levels


@pytest.mark.parametrize("path", ["zarr.json", "1/zarr.json"])
def test_empty_metadata_fails_cleanly(tmp_path, path):
    root = tmp_path / "image"
    write_image(root)
    (root / path).write_text("")
    with pytest.raises(damacy.InvalidArgument):
        load(root)


def test_native_query_validation(tmp_path):
    from damacy import _native

    root = tmp_path / "image"
    write_image(root)
    image = load(root)
    args = (image._native, (4, 4))
    with pytest.raises(ValueError):
        _native.spatial_resolve(*args, ((1, 0), (0, 1)), 1, 1, 0, -1)
    for filter, boundary, value, level in [
        (0, 1, 0, -1),
        (1, 0, 0, -1),
        (1, 1, 1, -1),
        (1, 2, float("nan"), -1),
        (1, 1, 0, -2),
        (1, 1, 0, 3),
    ]:
        with pytest.raises(_native.DamacyError):
            _native.spatial_resolve(
                *args, ((1, 0, 0), (0, 1, 0)), filter, boundary, value, level
            )


@pytest.mark.parametrize(
    "key,value",
    [("shape", "[16,16]"), ("data_type", '"float32"'), ("node_type", '"group"')],
)
def test_duplicate_array_fields_are_rejected(tmp_path, key, value):
    root = tmp_path / "image"
    write_image(root)
    path = root / "0" / "zarr.json"
    text = path.read_text()
    path.write_text(text[:-1] + f', "{key}": {value}' + "}")
    with pytest.raises(damacy.InvalidArgument):
        load(root)


def test_collapsed_3d_transforms_are_rejected(tmp_path):
    root = tmp_path / "image"
    write_image(root, shape=(16, 16, 16))
    image = load(root)
    shape = (2, 2, 2)
    rng = np.random.default_rng(832)
    for _ in range(128):
        matrix = rng.normal(size=(3, 3))
        matrix[2] = matrix[0] + matrix[1]
        q = damacy.SpatialQuery(
            output_to_reference=np.column_stack((matrix, [4, 4, 4])),
            sampler=damacy.Sampler(boundary="constant"),
        )
        with pytest.raises(damacy.InvalidArgument):
            image.resolve(q, shape=shape)


@pytest.mark.parametrize("spacing,level", [(0.5, 0), (3.0, 1)])
def test_level_selection_with_large_scale_difference(tmp_path, spacing, level):
    root = tmp_path / "image"
    write_image(root)
    image = load(root)
    shape = (2, 2)
    large = 2**28
    matrix = (
        (large + spacing / 2, large - spacing / 2, 0),
        (large - spacing / 2, large + spacing / 2, 0),
    )
    resolved = image.resolve(
        damacy.SpatialQuery(
            output_to_reference=matrix,
            sampler=damacy.Sampler(boundary="constant"),
        ),
        shape=shape,
    )
    assert resolved.level == level


@pytest.mark.parametrize("tail", [",", ", garbage", ', {"unused": [garbage,]}'])
def test_unselected_metadata_is_not_validated(tmp_path, tail):
    root = tmp_path / "image"
    write_image(root)
    path = root / "zarr.json"
    metadata = json.loads(path.read_text())
    entry = metadata["attributes"]["ome"]["multiscales"][0]
    entry["coordinateTransformations"] = {"unused": ["not", "interpreted"]}
    metadata["attributes"]["unused"] = {"arbitrary": {"nested": None}}
    text = json.dumps(metadata)
    position = text.rindex("]")
    path.write_text(text[:position] + tail + text[position:] + " ignored suffix")
    image = load(root)
    resolved = image.resolve(query(2), shape=(4, 4))
    assert resolved.level == 1
    assert not resolved.requires_resampling


def test_resolution_only_needs_output_shape(tmp_path):
    root = tmp_path / "image"
    write_image(root)
    image = load(root)
    shape = (1 << 40, 1 << 40)
    resolved = image.resolve(
        query(sampler=damacy.Sampler(boundary="constant")), shape=shape
    )
    assert resolved.shape == shape
    assert resolved.read_bounds_index == ((0, 32), (0, 32))
    assert resolved.requires_resampling
    for invalid in [(), (4,), (0, 4), (-1, 4), (1 << 52, 4), (1.5, 4)]:
        with pytest.raises((ValueError, TypeError)):
            image.resolve(query(), shape=invalid)
    with pytest.raises(TypeError, match=r"NgffImage\.resolve"):
        damacy.ResolvedSpatialQuery()
