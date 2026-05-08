"""Pytest coverage for the user-facing damacy Python surface (issue #18).

Mirrors the C-side test_damacy_caps coverage at the bindings layer plus
end-to-end push/pop and stats assertions specific to the Python wrapper.
Construction tests do not require a GPU; end-to-end / stats tests do —
they need CUDA + nvcomp + a real device, the same prereq as the C ctest
CUDA suite.
"""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
from pathlib import Path

import damacy
import pytest
from damacy import (
    BatchInfo,
    Config,
    DamacyError,
    DtypeMismatch,
    InvalidArgument,
    NotFound,
    OutOfMemory,
    Pipeline,
    Sample,
    Stats,
    Status,
    _native,
)


def _base_config(store_root: Path, dtype: str | int | damacy.Dtype = "f32") -> Config:
    """Minimum-viable Config for tests.

    Mirrors the defaults used by tests/test_damacy_caps.c::mk_cfg.
    `dtype` is the *destination* batch dtype (#16): F32 or BF16. Source
    zarrs cast in-kernel; tiny_zarr is u16 → f32 here.
    """
    return Config(
        store_root=str(store_root),
        batch_size=1,
        host_buffer_bytes=1 << 20,
        device_buffer_bytes=1 << 20,
        dtype=dtype,
        lookahead_batches=2,
        n_io_threads=1,
        n_zarrs_meta_cache=4,
        n_shards_meta_cache=4,
    )


# ---- module surface -----------------------------------------------------


def test_module_version_matches_native():
    assert damacy.__version__ == _native.__version__


def test_log_level_constants_aligned():
    assert damacy.LOG_TRACE == _native.LOG_TRACE == 0
    assert damacy.LOG_FATAL == _native.LOG_FATAL == 5


def test_status_enum_values_mirror_native():
    assert Status.OK == _native.STATUS_OK == 0
    assert Status.SHUTDOWN == _native.STATUS_SHUTDOWN
    # Round-trip a few; if the C side ever renumbers, this catches it.
    for s in (Status.AGAIN, Status.INVAL, Status.NOTFOUND, Status.DTYPE):
        assert Status(int(s)) is s


def test_dtype_coerce():
    assert damacy.Dtype.coerce("f32") is damacy.Dtype.F32
    assert damacy.Dtype.coerce("float32") is damacy.Dtype.F32
    assert damacy.Dtype.coerce("bf16") is damacy.Dtype.BF16
    assert damacy.Dtype.coerce("bfloat16") is damacy.Dtype.BF16
    assert damacy.Dtype.coerce(0) is damacy.Dtype.F32
    assert damacy.Dtype.coerce(damacy.Dtype.BF16) is damacy.Dtype.BF16
    with pytest.raises(ValueError, match="unknown dtype"):
        damacy.Dtype.coerce("u16")


# ---- construction & validation ------------------------------------------


def test_oversize_max_chunk_raises_invalid_argument(tmp_path):
    cfg = dataclasses.replace(
        _base_config(tmp_path),
        max_chunk_uncompressed_bytes=_native.MAX_CHUNK_UNCOMPRESSED_BYTES + 1,
    )
    with pytest.raises(InvalidArgument) as excinfo:
        Pipeline(cfg)
    # The base class is still _native.DamacyError so legacy `except
    # RuntimeError` catches keep working.
    assert excinfo.value.status is Status.INVAL
    assert isinstance(excinfo.value, _native.DamacyError)
    assert isinstance(excinfo.value, RuntimeError)


def test_max_gpu_memory_too_small_raises_oom(tmp_path):
    cfg = dataclasses.replace(_base_config(tmp_path), max_gpu_memory_bytes=64)
    with pytest.raises(OutOfMemory) as excinfo:
        Pipeline(cfg)
    assert excinfo.value.status is Status.OOM


@pytest.mark.parametrize("dtype", ["f32", "bf16", "float32", "bfloat16"])
def test_dtype_string_form_accepted(tiny_zarr, dtype):
    root, _ = tiny_zarr
    with Pipeline(_base_config(root, dtype=dtype)) as d:
        assert isinstance(d, Pipeline)
        assert d.config.dtype is damacy.Dtype.coerce(dtype)


def test_dtype_int_form_accepted(tiny_zarr):
    root, _ = tiny_zarr
    with Pipeline(_base_config(root, dtype=damacy.Dtype.BF16)) as d:
        assert d.config.dtype is damacy.Dtype.BF16


def test_dtype_unknown_string_raises(tmp_path):
    # Validation runs in Config.__post_init__ — fails before we touch CUDA.
    with pytest.raises(ValueError, match="unknown dtype"):
        _base_config(tmp_path, dtype="u16")


# ---- end-to-end push / pop / release ------------------------------------


def test_push_pop_release_via_context_managers(tiny_zarr):
    root, uri = tiny_zarr
    with Pipeline(_base_config(root)) as d:
        consumed = d.push([Sample(uri=uri, aabb=[(0, 8), (0, 16)])])
        assert consumed == 1

        with d.pop() as batch:
            info = batch.info
            assert isinstance(info, BatchInfo)
            assert info.shape == (1, 8, 16)
            assert info.dtype is damacy.Dtype.F32
            assert info.batch_id == 0
            assert info.device_ptr != 0
            assert "Batch" in repr(batch)


def test_unknown_uri_raises_notfound(tiny_zarr):
    root, _ = tiny_zarr
    with Pipeline(_base_config(root)) as d:
        with pytest.raises(NotFound) as excinfo:
            d.push([Sample(uri="not_a_zarr", aabb=[(0, 8), (0, 16)])])
        assert excinfo.value.status is Status.NOTFOUND


def test_unsupported_src_dtype_raises_dtype_mismatch(tiny_zarr_no_cast):
    root, uri = tiny_zarr_no_cast
    with Pipeline(_base_config(root, dtype="f32")) as d:
        with pytest.raises(DtypeMismatch) as excinfo:
            d.push([Sample(uri=uri, aabb=[(0, 8), (0, 16)])])
        assert excinfo.value.status is Status.DTYPE


def test_oversize_chunk_surfaces_at_pop(tmp_path, write_zarr_script):
    if not shutil.which("uv"):
        pytest.skip("uv not on PATH")
    out = tmp_path / "foo"
    cmd = [
        "uv", "run", "--script", str(write_zarr_script),
        "--out", str(out), "--shape", "8,16", "--inner", "8,16",
        "--shard", "8,16", "--dtype", "uint16", "--codec", "blosc-lz4",
    ]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"write_zarr.py failed: {r.stderr}")

    cfg = dataclasses.replace(
        _base_config(tmp_path),
        max_chunk_uncompressed_bytes=128,  # 8x16 u16 = 256 B → over
    )
    with Pipeline(cfg) as d:
        consumed = d.push([Sample(uri="foo", aabb=[(0, 8), (0, 16)])])
        assert consumed == 1
        with pytest.raises(InvalidArgument):
            d.pop()


# ---- batches() iterator -------------------------------------------------


def test_batches_iterator_yields_n(tiny_zarr):
    root, uri = tiny_zarr
    samples = [Sample(uri=uri, aabb=[(0, 8), (0, 16)]) for _ in range(3)]
    # Bump lookahead so all 3 batches fit in the user-push queue at once.
    cfg = dataclasses.replace(_base_config(root), lookahead_batches=4)
    with Pipeline(cfg) as d:
        consumed = d.push(samples)
        assert consumed == 3
        ids: list[int] = []
        for batch in d.batches(3):
            with batch as b:
                ids.append(b.info.batch_id)
        assert ids == [0, 1, 2]


# ---- Config dataclass --------------------------------------------------


def test_config_validates_eagerly():
    with pytest.raises(ValueError, match="batch_size"):
        Config(
            store_root="/tmp",
            batch_size=0,
            host_buffer_bytes=1 << 20,
            device_buffer_bytes=1 << 20,
        )
    with pytest.raises(ValueError, match="lookahead_batches"):
        Config(
            store_root="/tmp",
            batch_size=1,
            host_buffer_bytes=1 << 20,
            device_buffer_bytes=1 << 20,
            lookahead_batches=1,
        )
    with pytest.raises(ValueError, match="n_io_threads"):
        Config(
            store_root="/tmp",
            batch_size=1,
            host_buffer_bytes=1 << 20,
            device_buffer_bytes=1 << 20,
            n_io_threads=0,
        )
    with pytest.raises(ValueError, match="host/device_buffer_bytes"):
        Config(
            store_root="/tmp",
            batch_size=1,
            host_buffer_bytes=0,
            device_buffer_bytes=1 << 20,
        )
    with pytest.raises(ValueError, match="max_chunk_uncompressed_bytes"):
        Config(
            store_root="/tmp",
            batch_size=1,
            host_buffer_bytes=1 << 20,
            device_buffer_bytes=1 << 20,
            max_chunk_uncompressed_bytes=-1,
        )


def test_config_dtype_coerced(tmp_path):
    cfg = _base_config(tmp_path, dtype="bf16")
    assert cfg.dtype is damacy.Dtype.BF16


def test_config_replace_for_variants(tmp_path):
    base = _base_config(tmp_path)
    big = dataclasses.replace(base, batch_size=4)
    assert base.batch_size == 1 and big.batch_size == 4
    # frozen
    with pytest.raises(dataclasses.FrozenInstanceError):
        base.batch_size = 99  # type: ignore[misc]


# ---- stats --------------------------------------------------------------


def test_stats_returns_typed_dataclass(tiny_zarr):
    root, _ = tiny_zarr
    with Pipeline(_base_config(root)) as d:
        s = d.stats()
        assert isinstance(s, Stats)
        assert s.gpu_bytes_committed > 0
        assert s.plan.name == "plan"


def test_stats_gpu_bytes_grows_after_first_pop(tiny_zarr):
    root, uri = tiny_zarr
    with Pipeline(_base_config(root)) as d:
        before = d.stats().gpu_bytes_committed
        d.push([Sample(uri=uri, aabb=[(0, 8), (0, 16)])])
        with d.pop():
            pass
        after = d.stats().gpu_bytes_committed
        assert after > before


# ---- explicit flush / stats_reset --------------------------------------


def test_flush_and_stats_reset_are_idempotent(tiny_zarr):
    root, uri = tiny_zarr
    with Pipeline(_base_config(root)) as d:
        d.push([Sample(uri=uri, aabb=[(0, 8), (0, 16)])])
        d.flush()
        d.flush()  # idempotent
        with d.pop():
            pass
        s_before = d.stats()
        d.stats_reset()
        s_after = d.stats()
        # Most counters reset; gpu_bytes_committed reflects committed
        # buffers and stays put.
        assert s_after.batches_emitted == 0
        assert s_before.batches_emitted >= 1


# ---- DLPack export -----------------------------------------------------


def test_batch_dlpack_export_smoke(tiny_zarr):
    root, uri = tiny_zarr
    with Pipeline(_base_config(root)) as d:
        d.push([Sample(uri=uri, aabb=[(0, 8), (0, 16)])])
        with d.pop() as batch:
            # __dlpack_device__ → (kDLCUDA=2, ordinal).
            dev = batch.__dlpack_device__()
            assert dev[0] == 2
            # __dlpack__ returns a capsule; consuming it would need
            # torch/cupy. Leave the capsule unconsumed; the C-side
            # deleter is exercised when the capsule is GC'd.
            cap = batch.__dlpack__(stream=None)
            assert cap is not None
            del cap


# ---- log helpers -------------------------------------------------------


def test_set_log_level_and_quiet_passthrough():
    # Cycle through legal values; they forward into the C-side sink.
    damacy.set_log_level(damacy.LOG_DEBUG)
    damacy.set_log_level(damacy.LOG_INFO)
    damacy.set_log_quiet(True)
    damacy.set_log_quiet(False)


# ---- Batch repr after release ------------------------------------------


def test_batch_repr_handles_released_state(tiny_zarr):
    root, uri = tiny_zarr
    with Pipeline(_base_config(root)) as d:
        d.push([Sample(uri=uri, aabb=[(0, 8), (0, 16)])])
        b = d.pop()
        assert "Batch(batch_id=" in repr(b)
        b.release()
        assert "<released>" in repr(b)


# ---- closed-handle hygiene ---------------------------------------------


def test_use_after_close_raises(tiny_zarr):
    root, _ = tiny_zarr
    d = Pipeline(_base_config(root))
    d.close()
    d.close()  # idempotent
    with pytest.raises(AttributeError):
        d.stats()


# ---- exception hierarchy ------------------------------------------------


def test_per_status_exceptions_share_base(tmp_path):
    # Cheap construction-time error: oversize max_chunk → INVAL.
    cfg = dataclasses.replace(
        _base_config(tmp_path),
        max_chunk_uncompressed_bytes=_native.MAX_CHUNK_UNCOMPRESSED_BYTES + 1,
    )
    with pytest.raises(DamacyError) as excinfo:
        Pipeline(cfg)
    assert isinstance(excinfo.value, InvalidArgument)
    assert isinstance(excinfo.value, DamacyError)
    assert isinstance(excinfo.value, _native.DamacyError)
