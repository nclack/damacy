"""Pytest coverage for the user-facing damacy Python surface (issue #18).

Mirrors the C-side test_damacy_caps coverage at the bindings layer plus
end-to-end push/pop and stats assertions specific to the Python wrapper.
Construction tests do not require a GPU; end-to-end / stats tests do —
they need CUDA + nvcomp + a real device, the same prereq as the C ctest
CUDA suite.
"""

from __future__ import annotations

import ctypes
import dataclasses
import shutil
import subprocess
import sys
import warnings

import damacy
import pytest
from damacy import (
    BatchInfo,
    BudgetExceeded,
    Config,
    DamacyError,
    DtypeMismatch,
    InvalidArgument,
    NotFound,
    Pipeline,
    Sample,
    Stats,
    Status,
    _native,
)

# All tests in this file exercise the C extension end-to-end and need
# a primary CUDA context. Skip cleanly on CPU-only runners.
pytestmark = pytest.mark.usefixtures("cuda_ctx")


# ---- DLPack capsule layout (ctypes mirror of dmlc/dlpack) ----------------
#
# Lets the DLPack tests below inspect the actual capsule contents without
# depending on torch. Layouts match python/damacy/_api.c.

_kDLCUDA = 2
_kDLFloat = 2  # DLDataTypeCode.kDLFloat
_kDLBfloat = 4  # DLDataTypeCode.kDLBfloat


class _DLDevice(ctypes.Structure):
    _fields_ = [
        ("device_type", ctypes.c_int32),
        ("device_id", ctypes.c_int32),
    ]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


class _DLPackVersion(ctypes.Structure):
    _fields_ = [
        ("major", ctypes.c_uint32),
        ("minor", ctypes.c_uint32),
    ]


class _DLManagedTensorVersioned(ctypes.Structure):
    _fields_ = [
        ("version", _DLPackVersion),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
        ("flags", ctypes.c_uint64),
        ("dl_tensor", _DLTensor),
    ]


_PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_PyCapsule_GetPointer.restype = ctypes.c_void_p
_PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


def _capsule_as(cap, name: bytes, struct_t):
    """Read a capsule's payload pointer (without consuming it) and cast it
    to ``struct_t``. The capsule is unchanged — the consumer would rename
    it to ``used_<name>`` to mark consumption; we don't."""
    raw = _PyCapsule_GetPointer(cap, name)
    assert raw, "capsule had no pointer"
    return ctypes.cast(raw, ctypes.POINTER(struct_t)).contents


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
        n_array_meta_cache=4,
        n_shard_index_cache=4,
        n_chunk_layout_cache=4,
        sample_shape=(8, 16),
        max_gpu_memory_bytes=1 << 30,
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


def test_invalid_config_raises_invalid_argument():
    cfg = dataclasses.replace(_base_config(), n_array_meta_cache=0)
    with pytest.raises(InvalidArgument) as excinfo:
        Pipeline(cfg)
    # The base class is still _native.DamacyError so legacy `except
    # RuntimeError` catches keep working.
    assert excinfo.value.status is Status.INVAL
    assert isinstance(excinfo.value, _native.DamacyError)
    assert isinstance(excinfo.value, RuntimeError)


def test_max_gpu_memory_too_small_raises_budget():
    cfg = dataclasses.replace(_base_config(), max_gpu_memory_bytes=64)
    with pytest.raises(BudgetExceeded) as excinfo:
        Pipeline(cfg)
    assert excinfo.value.status is Status.BUDGET


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


@pytest.mark.skip(reason="PR-3 will update expected error placement (push → pop)")
def test_unknown_uri_raises_notfound(tiny_zarr):
    _ = tiny_zarr
    with Pipeline(_base_config()) as d:
        with pytest.raises(NotFound) as excinfo:
            d.push([Sample(uri="not_a_zarr", aabb=[(0, 8), (0, 16)])])
        assert excinfo.value.status is Status.NOTFOUND


@pytest.mark.skip(reason="PR-3 will update expected error placement (push → pop)")
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
        with pytest.raises(BudgetExceeded):
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


@pytest.mark.skip(reason="PR-3 will update expected error placement (push → pop)")
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
    ss = (8, 16)
    gpu = 1 << 30
    with pytest.raises(ValueError, match="batch_size"):
        Config(batch_size=0, sample_shape=ss, max_gpu_memory_bytes=gpu)
    with pytest.raises(ValueError, match="lookahead_batches"):
        Config(
            batch_size=1,
            sample_shape=ss,
            lookahead_batches=1,
            max_gpu_memory_bytes=gpu,
        )
    with pytest.raises(ValueError, match="n_io_threads"):
        Config(batch_size=1, sample_shape=ss, n_io_threads=0, max_gpu_memory_bytes=gpu)
    with pytest.raises(ValueError, match="max_chunk_uncompressed_bytes"):
        Config(
            batch_size=1,
            sample_shape=ss,
            max_chunk_uncompressed_bytes=-1,
            max_gpu_memory_bytes=gpu,
        )
    with pytest.raises(ValueError, match="max_gpu_memory_bytes"):
        Config(batch_size=1, sample_shape=ss, max_gpu_memory_bytes=0)
    with pytest.raises(ValueError, match="sample_shape"):
        Config(batch_size=1, sample_shape=(), max_gpu_memory_bytes=gpu)
    with pytest.raises(ValueError, match="sample_shape"):
        Config(batch_size=1, sample_shape=(8, 0), max_gpu_memory_bytes=gpu)
    with pytest.raises(ValueError, match="max_read_op_bytes"):
        Config(
            batch_size=1,
            sample_shape=ss,
            max_read_op_bytes=-1,
            max_gpu_memory_bytes=gpu,
        )
    with pytest.raises(ValueError, match="numa_node"):
        Config(
            batch_size=1,
            sample_shape=ss,
            max_gpu_memory_bytes=gpu,
            numa_strategy="pin_to",
        )
    with pytest.raises(ValueError, match="numa_node"):
        Config(
            batch_size=1,
            sample_shape=ss,
            max_gpu_memory_bytes=gpu,
            numa_strategy="auto",
            numa_node=3,
        )


def test_config_numa_defaults_and_coercion():
    cfg = _base_config()
    assert cfg.numa_strategy is damacy.NumaStrategy.AUTO
    assert cfg.numa_node == -1
    cfg2 = dataclasses.replace(cfg, numa_strategy=damacy.NumaStrategy.DISABLED)
    assert cfg2.numa_strategy is damacy.NumaStrategy.DISABLED
    cfg3 = dataclasses.replace(cfg, numa_strategy="pin_to", numa_node=0)
    assert cfg3.numa_strategy is damacy.NumaStrategy.PIN_TO
    assert cfg3.numa_node == 0


def test_config_enable_gds_tri_state():
    cfg = _base_config()
    assert cfg.enable_gds is None  # AUTO default
    assert dataclasses.replace(cfg, enable_gds=True).enable_gds is True
    assert dataclasses.replace(cfg, enable_gds=False).enable_gds is False


def test_pipeline_accepts_new_config_kwargs(tiny_zarr):
    _ = tiny_zarr
    cfg = dataclasses.replace(
        _base_config(),
        enable_gds=False,
        numa_strategy=damacy.NumaStrategy.DISABLED,
    )
    with Pipeline(cfg) as d:
        assert d is not None


def test_pipeline_explicit_gds_off_overrides_env(tiny_zarr, monkeypatch):
    _ = tiny_zarr
    monkeypatch.setenv("DAMACY_GDS_ENABLE", "1")
    cfg = dataclasses.replace(_base_config(), enable_gds=False)
    with Pipeline(cfg) as d:
        assert d is not None


def test_native_pipeline_rejects_out_of_range_enums(tiny_zarr):
    _ = tiny_zarr

    def _build(**override):
        return _native.Pipeline(
            batch_size=1,
            lookahead_batches=2,
            n_io_threads=1,
            n_array_meta_cache=4,
            n_shard_index_cache=4,
            n_chunk_layout_cache=4,
            dtype=_native.DTYPE_F32,
            max_chunk_uncompressed_bytes=0,
            max_gpu_memory_bytes=1 << 30,
            sample_shape=(8, 16),
            **override,
        )

    with pytest.raises(ValueError, match="enable_gds"):
        _build(enable_gds=99)
    with pytest.raises(ValueError, match="numa_strategy"):
        _build(numa_strategy=99)


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


def test_stats_io_planner_counters_advance(tiny_zarr):
    uri = tiny_zarr
    with Pipeline(_base_config()) as d:
        d.push([Sample(uri=uri, aabb=[(0, 8), (0, 16)])])
        with d.pop():
            pass
        s = d.stats()
        assert s.chunks_planned > 0
        assert s.chunks_to_load > 0
        assert s.reads_issued > 0
        assert s.chunks_to_load <= s.chunks_planned


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


def test_batch_dlpack_default_is_v0_capsule(tiny_zarr):
    """Default __dlpack__() (no max_version) must emit a v0 capsule —
    that's the only thing PyTorch 2.8 consumes (issue #33)."""
    uri = tiny_zarr
    with Pipeline(_base_config()) as d:
        d.push([Sample(uri=uri, aabb=[(0, 8), (0, 16)])])
        with d.pop() as batch:
            cap = batch.__dlpack__(stream=None)
            # PyCapsule.__repr__ embeds the name, e.g.
            #   '<capsule object "dltensor" at 0x...>'
            assert 'dltensor"' in repr(cap)
            assert "dltensor_versioned" not in repr(cap)
            del cap


def test_batch_dlpack_versioned_when_max_version_requests_it(tiny_zarr):
    """max_version=(1, 0) opts the consumer into the v1.0 wire format."""
    uri = tiny_zarr
    with Pipeline(_base_config()) as d:
        d.push([Sample(uri=uri, aabb=[(0, 8), (0, 16)])])
        with d.pop() as batch:
            cap = batch.__dlpack__(stream=None, max_version=(1, 0))
            assert "dltensor_versioned" in repr(cap)
            del cap


def test_batch_dlpack_max_version_zero_stays_v0(tiny_zarr):
    """max_version=(0, X) is a legacy-only request → v0."""
    uri = tiny_zarr
    with Pipeline(_base_config()) as d:
        d.push([Sample(uri=uri, aabb=[(0, 8), (0, 16)])])
        with d.pop() as batch:
            cap = batch.__dlpack__(stream=None, max_version=(0, 9))
            assert 'dltensor"' in repr(cap)
            assert "dltensor_versioned" not in repr(cap)
            del cap


def test_batch_torch_from_dlpack(tiny_zarr):
    """torch.from_dlpack(batch) must accept the default capsule — that's
    the regression issue #33 exists to fix."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-enabled torch required")

    uri = tiny_zarr
    with Pipeline(_base_config()) as d:
        d.push([Sample(uri=uri, aabb=[(0, 8), (0, 16)])])
        with d.pop() as batch:
            info = batch.info
            t = torch.from_dlpack(batch)
            assert t.device.type == "cuda"
            assert tuple(t.shape) == tuple(info.shape)
            assert t.dtype == torch.float32  # _base_config default
            # The data should be readable. Copy off-device and check it
            # isn't garbage / segfaulting.
            host = t.detach().cpu()
            assert host.numel() == t.numel()


def test_batch_torch_from_dlpack_versioned(tiny_zarr):
    """torch.utils.dlpack.from_dlpack should also accept the v1 capsule
    when a caller explicitly asks for it. PyTorch 2.6+ understands both
    names, so this exercises the v1 path end-to-end."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-enabled torch required")
    # Some PyTorch builds (<2.6) reject "dltensor_versioned" outright;
    # treat that as a known limitation and skip rather than fail.
    uri = tiny_zarr
    with Pipeline(_base_config()) as d:
        d.push([Sample(uri=uri, aabb=[(0, 8), (0, 16)])])
        with d.pop() as batch:
            info = batch.info
            cap = batch.__dlpack__(stream=None, max_version=(1, 0))
            try:
                t = torch.utils.dlpack.from_dlpack(cap)
            except (RuntimeError, TypeError) as exc:
                pytest.skip(f"torch doesn't accept v1 capsule: {exc}")
            assert t.device.type == "cuda"
            assert tuple(t.shape) == tuple(info.shape)


def test_batch_dlpack_v0_capsule_fields(tiny_zarr):
    """Parse the v0 DLManagedTensor and verify every field matches what
    `batch.info` reports: data ptr, device, ndim, dtype, shape, no
    strides (contiguous), zero byte_offset."""
    uri = tiny_zarr
    with Pipeline(_base_config()) as d:
        d.push([Sample(uri=uri, aabb=[(0, 8), (0, 16)])])
        with d.pop() as batch:
            info = batch.info
            cap = batch.__dlpack__(stream=None)
            mt = _capsule_as(cap, b"dltensor", _DLManagedTensor)
            t = mt.dl_tensor
            assert t.data == info.device_ptr
            assert t.device.device_type == _kDLCUDA
            assert t.ndim == len(info.shape)
            assert t.dtype.code == _kDLFloat  # f32 from _base_config
            assert t.dtype.bits == 32
            assert t.dtype.lanes == 1
            assert [t.shape[i] for i in range(t.ndim)] == list(info.shape)
            # damacy emits contiguous tensors, so strides is NULL.
            assert not t.strides
            assert t.byte_offset == 0
            # The deleter is wired (non-NULL); manager_ctx points at
            # damacy's payload struct, also non-NULL.
            assert mt.deleter
            assert mt.manager_ctx
            del cap


def test_batch_dlpack_v1_capsule_fields(tiny_zarr):
    """Same field-level check on the versioned capsule. Version=(1,0),
    flags=0, dl_tensor mirrors v0."""
    uri = tiny_zarr
    with Pipeline(_base_config()) as d:
        d.push([Sample(uri=uri, aabb=[(0, 8), (0, 16)])])
        with d.pop() as batch:
            info = batch.info
            cap = batch.__dlpack__(stream=None, max_version=(1, 0))
            mt = _capsule_as(cap, b"dltensor_versioned", _DLManagedTensorVersioned)
            assert mt.version.major == 1
            assert mt.version.minor == 0
            assert mt.flags == 0
            t = mt.dl_tensor
            assert t.data == info.device_ptr
            assert t.device.device_type == _kDLCUDA
            assert t.ndim == len(info.shape)
            assert [t.shape[i] for i in range(t.ndim)] == list(info.shape)
            assert mt.deleter
            assert mt.manager_ctx
            del cap


def test_batch_dlpack_v0_capsule_fields_bf16(tiny_zarr):
    """Same as v0_capsule_fields but with destination dtype=bf16 — the
    DLDataType must report code=kDLBfloat (4), bits=16, lanes=1."""
    uri = tiny_zarr
    with Pipeline(_base_config(dtype="bf16")) as d:
        d.push([Sample(uri=uri, aabb=[(0, 8), (0, 16)])])
        with d.pop() as batch:
            cap = batch.__dlpack__(stream=None)
            mt = _capsule_as(cap, b"dltensor", _DLManagedTensor)
            t = mt.dl_tensor
            assert t.dtype.code == _kDLBfloat
            assert t.dtype.bits == 16
            assert t.dtype.lanes == 1
            del cap


def test_batch_dlpack_copy_true_rejected(tiny_zarr):
    """Per DLPack protocol, copy=True asks the producer to materialize
    a fresh buffer. damacy doesn't support that — it must raise
    BufferError so the consumer falls back."""
    uri = tiny_zarr
    with Pipeline(_base_config()) as d:
        d.push([Sample(uri=uri, aabb=[(0, 8), (0, 16)])])
        with d.pop() as batch:
            with pytest.raises(BufferError, match="copy=True not supported"):
                batch.__dlpack__(stream=None, copy=True)
            # copy=False and copy=None remain fine.
            cap1 = batch.__dlpack__(stream=None, copy=False)
            cap2 = batch.__dlpack__(stream=None, copy=None)
            del cap1, cap2


def test_batch_dlpack_invalid_max_version(tiny_zarr):
    """max_version must be either None or a 2-element non-negative
    sequence. Anything else raises TypeError/ValueError cleanly."""
    uri = tiny_zarr
    with Pipeline(_base_config()) as d:
        d.push([Sample(uri=uri, aabb=[(0, 8), (0, 16)])])
        with d.pop() as batch:
            # wrong arity
            with pytest.raises(TypeError, match="2-element"):
                batch.__dlpack__(stream=None, max_version=(1,))  # type: ignore[arg-type]
            with pytest.raises(TypeError, match="2-element"):
                batch.__dlpack__(stream=None, max_version=(1, 0, 0))  # type: ignore[arg-type]
            # negative components
            with pytest.raises(ValueError, match=">= 0"):
                batch.__dlpack__(stream=None, max_version=(-1, 0))
            # non-integer entries
            with pytest.raises(TypeError):
                batch.__dlpack__(stream=None, max_version=("1", "0"))  # type: ignore[arg-type]


def test_batch_dlpack_after_release_raises(tiny_zarr):
    """Both __dlpack__ and __dlpack_device__ must fail with RuntimeError
    once the Batch has been released — no use-after-free into damacy."""
    uri = tiny_zarr
    with Pipeline(_base_config()) as d:
        d.push([Sample(uri=uri, aabb=[(0, 8), (0, 16)])])
        batch = d.pop()
        batch.release()
        with pytest.raises(RuntimeError, match="released"):
            batch.__dlpack__(stream=None)
        with pytest.raises(RuntimeError, match="released"):
            batch.__dlpack_device__()


def test_batch_dlpack_capsule_holds_batch_alive(tiny_zarr):
    """The capsule's Py_INCREF lands on the C-side BatchObj
    (``batch._native``), not on the Python-level wrapper — the wrapper
    just delegates. Dropping the capsule must run the deleter and
    decrement that native reference back to its baseline."""
    uri = tiny_zarr
    with Pipeline(_base_config()) as d:
        d.push([Sample(uri=uri, aabb=[(0, 8), (0, 16)])])
        batch = d.pop()
        try:
            native = batch._native
            rc_before = sys.getrefcount(native)
            cap = batch.__dlpack__(stream=None)
            rc_held = sys.getrefcount(native)
            assert rc_held > rc_before, "capsule should incref the native batch"
            del cap
            rc_after = sys.getrefcount(native)
            assert rc_after == rc_before, "capsule deleter should decref it"
        finally:
            batch.release()


def test_batch_dlpack_stream_kwargs_accepted(tiny_zarr):
    """The DLPack protocol stream sentinel values must all be accepted:
    None (producer sync), -1 (no sync), 0/1/2 (legacy / default streams),
    arbitrary int (consumer stream handle). The last one is best-effort
    — we just verify the call doesn't raise."""
    uri = tiny_zarr
    with Pipeline(_base_config()) as d:
        d.push([Sample(uri=uri, aabb=[(0, 8), (0, 16)])])
        with d.pop() as batch:
            for s in (None, -1, 0, 1, 2):
                cap = batch.__dlpack__(stream=s)
                assert cap is not None
                del cap


def test_batch_dlpack_device_returns_pipeline_device(tiny_zarr):
    """__dlpack_device__ must report (kDLCUDA, device_ord) where
    device_ord matches the Pipeline's bound device."""
    uri = tiny_zarr
    with Pipeline(_base_config()) as d:
        d.push([Sample(uri=uri, aabb=[(0, 8), (0, 16)])])
        with d.pop() as batch:
            dev_type, dev_id = batch.__dlpack_device__()
            assert dev_type == _kDLCUDA
            assert dev_id == d.device


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
    # Cheap construction-time INVAL.
    cfg = dataclasses.replace(_base_config(), n_array_meta_cache=0)
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


def test_multi_gpu_implicit_warns(tiny_zarr, monkeypatch):
    _ = tiny_zarr
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    # Force the multi-GPU branch even on single-GPU CI by faking the
    # device-count probe; the heuristic should fire on any (count > 1).
    monkeypatch.setattr(_native, "cuda_device_count", lambda: 4)
    damacy._warned_multi_gpu_pairs.clear()
    with pytest.warns(UserWarning, match=r"device 0 of 4"):
        Pipeline(_base_config()).close()


def test_multi_gpu_warning_dedupes(tiny_zarr, monkeypatch):
    _ = tiny_zarr
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(_native, "cuda_device_count", lambda: 4)
    damacy._warned_multi_gpu_pairs.clear()
    with pytest.warns(UserWarning, match=r"device 0 of 4"):
        Pipeline(_base_config()).close()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Pipeline(_base_config()).close()
    assert not any("of 4" in str(w.message) for w in caught)


def test_multi_gpu_quiet_on_single_gpu(tiny_zarr, monkeypatch, recwarn):
    _ = tiny_zarr
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(_native, "cuda_device_count", lambda: 1)
    Pipeline(_base_config()).close()
    ours = [w for w in recwarn.list if "of 1" in str(w.message)]
    assert ours == []


def test_multi_gpu_quiet_when_explicit_device(tiny_zarr, monkeypatch, recwarn):
    _ = tiny_zarr
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(_native, "cuda_device_count", lambda: 4)
    damacy._warned_multi_gpu_pairs.clear()
    cfg = dataclasses.replace(_base_config(), device=0)
    Pipeline(cfg).close()
    ours = [w for w in recwarn.list if "of 4" in str(w.message)]
    assert ours == []


def test_multi_gpu_suppressed_when_local_rank_fires(tiny_zarr, monkeypatch):
    _ = tiny_zarr
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(_native, "cuda_device_count", lambda: 4)
    damacy._warned_local_rank_pairs.clear()
    damacy._warned_multi_gpu_pairs.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Pipeline(_base_config()).close()
    assert any("LOCAL_RANK=3" in str(w.message) for w in caught)
    assert not any("of 4" in str(w.message) for w in caught)
