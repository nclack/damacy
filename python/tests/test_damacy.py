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
import warnings

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


def _base_config(dtype: str | int | damacy.Dtype = "f32") -> Config:
    """Minimum-viable Config for tests.

    Mirrors the defaults used by tests/test_damacy_caps.c::mk_cfg.
    `dtype` is the *destination* batch dtype (#16): F32 or BF16. Source
    zarrs cast in-kernel; tiny_zarr is u16 → f32 here.
    """
    return Config(
        batch_size=1,
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


def test_oversize_max_chunk_raises_invalid_argument():
    cfg = dataclasses.replace(
        _base_config(),
        max_chunk_uncompressed_bytes=_native.MAX_CHUNK_UNCOMPRESSED_BYTES + 1,
    )
    with pytest.raises(InvalidArgument) as excinfo:
        Pipeline(cfg)
    # The base class is still _native.DamacyError so legacy `except
    # RuntimeError` catches keep working.
    assert excinfo.value.status is Status.INVAL
    assert isinstance(excinfo.value, _native.DamacyError)
    assert isinstance(excinfo.value, RuntimeError)


def test_max_gpu_memory_too_small_raises_oom():
    cfg = dataclasses.replace(_base_config(), max_gpu_memory_bytes=64)
    with pytest.raises(OutOfMemory) as excinfo:
        Pipeline(cfg)
    assert excinfo.value.status is Status.OOM


@pytest.mark.parametrize("dtype", ["f32", "bf16", "float32", "bfloat16"])
def test_dtype_string_form_accepted(tiny_zarr, dtype):
    _ = tiny_zarr
    with Pipeline(_base_config(dtype=dtype)) as d:
        assert isinstance(d, Pipeline)
        assert d.config.dtype is damacy.Dtype.coerce(dtype)


def test_dtype_int_form_accepted(tiny_zarr):
    _ = tiny_zarr
    with Pipeline(_base_config(dtype=damacy.Dtype.BF16)) as d:
        assert d.config.dtype is damacy.Dtype.BF16


def test_dtype_unknown_string_raises():
    # Validation runs in Config.__init__ — fails before we touch CUDA.
    with pytest.raises(ValueError, match="unknown dtype"):
        _base_config(dtype="u16")


# ---- end-to-end push / pop / release ------------------------------------


def test_push_pop_release_via_context_managers(tiny_zarr):
    uri = tiny_zarr
    with Pipeline(_base_config()) as d:
        d.push([Sample(uri=uri, aabb=[(0, 8), (0, 16)])])

        with d.pop() as batch:
            info = batch.info
            assert isinstance(info, BatchInfo)
            assert info.shape == (1, 8, 16)
            assert info.dtype is damacy.Dtype.F32
            assert info.batch_id == 0
            assert info.device_ptr != 0
            assert "Batch" in repr(batch)


def test_unknown_uri_raises_notfound(tiny_zarr):
    _ = tiny_zarr
    with Pipeline(_base_config()) as d:
        with pytest.raises(NotFound) as excinfo:
            d.push([Sample(uri="not_a_zarr", aabb=[(0, 8), (0, 16)])])
        assert excinfo.value.status is Status.NOTFOUND


def test_unsupported_src_dtype_raises_dtype_mismatch(tiny_zarr_no_cast):
    uri = tiny_zarr_no_cast
    with Pipeline(_base_config(dtype="f32")) as d:
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
        "--shard", "8,16", "--dtype", "uint16", "--codec", "blosc-zstd",
    ]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"write_zarr.py failed: {r.stderr}")

    cfg = dataclasses.replace(
        _base_config(),
        max_chunk_uncompressed_bytes=128,  # 8x16 u16 = 256 B → over
    )
    with Pipeline(cfg) as d:
        d.push([Sample(uri=str(out), aabb=[(0, 8), (0, 16)])])
        with pytest.raises(InvalidArgument):
            d.pop()


# ---- batches() iterator -------------------------------------------------


def test_batches_iterator_yields_n(tiny_zarr):
    uri = tiny_zarr
    samples = [Sample(uri=uri, aabb=[(0, 8), (0, 16)]) for _ in range(3)]
    # Default lookahead (= 2 with batch_size=1, capacity=2) is now smaller
    # than the push; the wrapper queues the overflow and drains as we pop.
    with Pipeline(_base_config()) as d:
        d.push(samples)
        ids: list[int] = []
        for batch in d.batches(3):
            with batch as b:
                ids.append(b.info.batch_id)
        assert ids == [0, 1, 2]


def test_push_returns_none(tiny_zarr):
    uri = tiny_zarr
    with Pipeline(_base_config()) as d:
        assert d.push([Sample(uri=uri, aabb=[(0, 8), (0, 16)])]) is None


def _drain_ids(d: Pipeline, n: int) -> list[int]:
    ids: list[int] = []
    for batch in d.batches(n):
        with batch as b:
            ids.append(b.info.batch_id)
    return ids


def test_push_overflows_queue_into_pending(tiny_zarr):
    """Pushing more samples than the native lookahead holds is silently
    handled by the Python-side queue; pop drives the drain."""
    uri = tiny_zarr
    # default lookahead=2, batch_size=1 → capacity=2; we push 5.
    samples = [Sample(uri=uri, aabb=[(0, 8), (0, 16)]) for _ in range(5)]
    with Pipeline(_base_config()) as d:
        d.push(samples)
        assert d.pending  # 3 samples > capacity sitting in the deque
        assert _drain_ids(d, 5) == [0, 1, 2, 3, 4]
        assert not d.pending


def test_push_accepts_generator(tiny_zarr):
    uri = tiny_zarr

    def gen():
        for _ in range(4):
            yield Sample(uri=uri, aabb=[(0, 8), (0, 16)])

    with Pipeline(_base_config()) as d:
        d.push(gen())  # generator, not a list
        assert _drain_ids(d, 4) == [0, 1, 2, 3]


def test_push_accepts_infinite_generator(tiny_zarr):
    """Infinite generator must not be materialized; we pull only what fits
    and what each pop frees."""
    uri = tiny_zarr

    def forever():
        while True:
            yield Sample(uri=uri, aabb=[(0, 8), (0, 16)])

    with Pipeline(_base_config()) as d:
        d.push(forever())  # would OOM if eagerly materialized
        assert _drain_ids(d, 3) == [0, 1, 2]
        assert d.pending  # generator still has more


def test_push_chains_multiple_calls(tiny_zarr):
    uri = tiny_zarr
    s = Sample(uri=uri, aabb=[(0, 8), (0, 16)])
    with Pipeline(_base_config()) as d:
        d.push([s, s])
        d.push([s, s, s])
        assert _drain_ids(d, 5) == [0, 1, 2, 3, 4]


def test_push_error_drops_offending_iterator(tiny_zarr):
    """A NotFound surfaces the error and discards the failing iterator's
    tail. Samples accepted by earlier ``push`` calls (already in the
    native lookahead) still resolve through ``pop``."""
    uri = tiny_zarr
    good = Sample(uri=uri, aabb=[(0, 8), (0, 16)])
    bad = Sample(uri="not_a_zarr", aabb=[(0, 8), (0, 16)])
    with Pipeline(_base_config()) as d:
        d.push([good])  # drains synchronously into native
        with pytest.raises(NotFound):
            d.push([bad, good])  # raises on bad; the trailing good is dropped
        # The first push's sample is in the lookahead and still pops.
        with d.pop() as b:
            assert b.info.batch_id == 0


# ---- Config dataclass --------------------------------------------------


def test_config_validates_eagerly():
    with pytest.raises(ValueError, match="batch_size"):
        Config(batch_size=0)
    with pytest.raises(ValueError, match="lookahead_batches"):
        Config(batch_size=1, lookahead_batches=1)
    with pytest.raises(ValueError, match="n_io_threads"):
        Config(batch_size=1, n_io_threads=0)
    with pytest.raises(ValueError, match="host/device_buffer_bytes"):
        Config(batch_size=1, host_buffer_bytes=-1)
    with pytest.raises(ValueError, match="max_chunk_uncompressed_bytes"):
        Config(batch_size=1, max_chunk_uncompressed_bytes=-1)


def test_host_buffer_bytes_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="deprecated"):
        Config(batch_size=8, max_gpu_memory_bytes=1 << 30, host_buffer_bytes=1)


def test_device_buffer_bytes_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="deprecated"):
        Config(batch_size=8, max_gpu_memory_bytes=1 << 30, device_buffer_bytes=1)


def test_no_deprecation_warning_at_default_zero(recwarn):
    # Both deprecated knobs left at 0 (the default) — no DeprecationWarning
    # should fire from Config construction.
    Config(batch_size=8, max_gpu_memory_bytes=1 << 30)
    ours = [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]
    assert ours == []


def test_config_dtype_coerced():
    cfg = _base_config(dtype="bf16")
    assert cfg.dtype is damacy.Dtype.BF16


def test_config_replace_for_variants():
    base = _base_config()
    big = dataclasses.replace(base, batch_size=4)
    assert base.batch_size == 1 and big.batch_size == 4
    # frozen
    with pytest.raises(dataclasses.FrozenInstanceError):
        base.batch_size = 99  # type: ignore[misc]


# ---- stats --------------------------------------------------------------


def test_stats_returns_typed_dataclass(tiny_zarr):
    _ = tiny_zarr
    with Pipeline(_base_config()) as d:
        s = d.stats()
        assert isinstance(s, Stats)
        assert s.gpu_bytes_committed > 0
        assert s.plan.name == "plan"


def test_stats_gpu_bytes_grows_after_first_pop(tiny_zarr):
    uri = tiny_zarr
    with Pipeline(_base_config()) as d:
        before = d.stats().gpu_bytes_committed
        d.push([Sample(uri=uri, aabb=[(0, 8), (0, 16)])])
        with d.pop():
            pass
        after = d.stats().gpu_bytes_committed
        assert after > before


# ---- explicit flush / stats_reset --------------------------------------


def test_flush_and_stats_reset_are_idempotent(tiny_zarr):
    uri = tiny_zarr
    with Pipeline(_base_config()) as d:
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
    uri = tiny_zarr
    with Pipeline(_base_config()) as d:
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
    uri = tiny_zarr
    with Pipeline(_base_config()) as d:
        d.push([Sample(uri=uri, aabb=[(0, 8), (0, 16)])])
        b = d.pop()
        assert "Batch(batch_id=" in repr(b)
        b.release()
        assert "<released>" in repr(b)


# ---- closed-handle hygiene ---------------------------------------------


def test_use_after_close_raises(tiny_zarr):
    _ = tiny_zarr
    d = Pipeline(_base_config())
    d.close()
    d.close()  # idempotent
    # Every public method that touches the native handle surfaces a
    # typed ShutdownError, not a bare AttributeError.
    with pytest.raises(damacy.ShutdownError) as excinfo:
        d.stats()
    assert excinfo.value.status is Status.SHUTDOWN
    with pytest.raises(damacy.ShutdownError):
        d.pop()
    with pytest.raises(damacy.ShutdownError):
        d.push([Sample(uri="x", aabb=[(0, 1), (0, 1)])])
    with pytest.raises(damacy.ShutdownError):
        d.flush()
    with pytest.raises(damacy.ShutdownError):
        d.stats_reset()
    with pytest.raises(damacy.ShutdownError):
        _ = d.device


# ---- exception hierarchy ------------------------------------------------


def test_per_status_exceptions_share_base():
    # Cheap construction-time error: oversize max_chunk → INVAL.
    cfg = dataclasses.replace(
        _base_config(),
        max_chunk_uncompressed_bytes=_native.MAX_CHUNK_UNCOMPRESSED_BYTES + 1,
    )
    with pytest.raises(DamacyError) as excinfo:
        Pipeline(cfg)
    assert isinstance(excinfo.value, InvalidArgument)
    assert isinstance(excinfo.value, DamacyError)
    assert isinstance(excinfo.value, _native.DamacyError)


# ---- device binding -----------------------------------------------------


def test_pipeline_exposes_bound_device(tiny_zarr):
    _ = tiny_zarr
    with Pipeline(_base_config()) as p:
        assert p.device == 0  # pytest fixture binds dev 0


def test_explicit_device_zero_succeeds(tiny_zarr):
    _ = tiny_zarr
    cfg = dataclasses.replace(_base_config(), device=0)
    with Pipeline(cfg) as p:
        assert p.device == 0


def test_local_rank_disagreement_warns(tiny_zarr, monkeypatch):
    _ = tiny_zarr
    monkeypatch.setenv("LOCAL_RANK", "3")  # bound dev is 0; rank claims 3
    # Per-process dedup means a previous test could have absorbed the
    # warning for this (rank, bound) pair. Clear the cache so the
    # assertion is deterministic regardless of test order.
    damacy._warned_local_rank_pairs.clear()
    with pytest.warns(UserWarning, match="LOCAL_RANK=3"):
        Pipeline(_base_config()).close()


def test_local_rank_warning_dedupes(tiny_zarr, monkeypatch):
    _ = tiny_zarr
    monkeypatch.setenv("LOCAL_RANK", "3")
    damacy._warned_local_rank_pairs.clear()
    with pytest.warns(UserWarning, match="LOCAL_RANK=3"):
        Pipeline(_base_config()).close()
    # Second construction with the same misconfiguration is silent.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Pipeline(_base_config()).close()
    assert not any("LOCAL_RANK" in str(w.message) for w in caught)


def test_local_rank_match_is_quiet(tiny_zarr, monkeypatch, recwarn):
    _ = tiny_zarr
    monkeypatch.setenv("LOCAL_RANK", "0")
    Pipeline(_base_config()).close()
    # No UserWarning from our heuristic; other libs may warn — filter ours.
    ours = [w for w in recwarn.list if "LOCAL_RANK" in str(w.message)]
    assert ours == []


def test_explicit_device_suppresses_local_rank_warning(tiny_zarr, monkeypatch, recwarn):
    _ = tiny_zarr
    monkeypatch.setenv("LOCAL_RANK", "3")
    cfg = dataclasses.replace(_base_config(), device=0)
    Pipeline(cfg).close()
    ours = [w for w in recwarn.list if "LOCAL_RANK" in str(w.message)]
    assert ours == []


def test_local_rank_non_int_is_quiet(tiny_zarr, monkeypatch, recwarn):
    _ = tiny_zarr
    monkeypatch.setenv("LOCAL_RANK", "not-a-number")
    Pipeline(_base_config()).close()
    ours = [w for w in recwarn.list if "LOCAL_RANK" in str(w.message)]
    assert ours == []
