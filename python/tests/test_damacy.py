# Pytest coverage for the damacy._native bindings (issue #10).
#
# Mirrors the C-side test_damacy_caps coverage at the bindings layer,
# plus end-to-end push/pop and stats assertions specific to the Python
# wrapper. Construction tests do not require a GPU; end-to-end / stats
# tests do — they need CUDA + nvcomp + a real device, the same prereq
# as the C ctest CUDA suite.
from __future__ import annotations

import logging
from pathlib import Path

import pytest

import damacy
from damacy import _native


def _base_kwargs(store_root: Path, dtype: str | int = "f32") -> dict:
    """Minimum-viable kwargs for damacy._native.Damacy(...).

    Mirrors the defaults used by tests/test_damacy_caps.c::mk_cfg.
    `dtype` is the *destination* batch dtype (#16): F32 or BF16. Source
    zarrs cast in-kernel; tiny_zarr is u16 → f32 here.
    max_chunk_uncompressed_bytes is required by the binding; 0 means
    "use the C default" (512 KB), which is what the C test passes.
    """
    return {
        "store_root": str(store_root),
        "batch_size": 1,
        "lookahead_batches": 2,
        "n_io_threads": 1,
        "host_buffer_bytes": 1 << 20,
        "device_buffer_bytes": 1 << 20,
        "n_zarrs_meta_cache": 4,
        "n_shards_meta_cache": 4,
        "dtype": dtype,
        "max_chunk_uncompressed_bytes": 0,
    }


# ---------- module surface ----------

def test_module_constants():
    assert _native.MAX_CHUNK_UNCOMPRESSED_BYTES == (2 << 20)
    # Log-level constants stay aligned with damacy_log.h.
    assert _native.LOG_TRACE == 0
    assert _native.LOG_FATAL == 5


def test_module_version_matches_package():
    assert damacy.__version__ == _native.__version__


# ---------- construction & validation (no GPU work yet at this point) ----------

def test_missing_max_chunk_raises_type_error(tmp_path):
    kw = _base_kwargs(tmp_path)
    kw.pop("max_chunk_uncompressed_bytes")
    with pytest.raises(TypeError):
        _native.Damacy(**kw)


def test_oversize_max_chunk_rejected(tmp_path):
    # > DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES is rejected at create with
    # DAMACY_INVAL → RuntimeError surfacing the C status string.
    kw = _base_kwargs(tmp_path)
    kw["max_chunk_uncompressed_bytes"] = _native.MAX_CHUNK_UNCOMPRESSED_BYTES + 1
    with pytest.raises(RuntimeError, match="invalid argument"):
        _native.Damacy(**kw)


def test_max_gpu_memory_too_small_rejected(tmp_path):
    # Mirrors test_damacy_caps.c::test_gpu_budget_too_small — wave-resident
    # memory at default config is many MB, so create returns DAMACY_OOM.
    kw = _base_kwargs(tmp_path)
    kw["max_gpu_memory_bytes"] = 64
    with pytest.raises(RuntimeError, match="OOM|out of memory"):
        _native.Damacy(**kw)


@pytest.mark.parametrize("dtype", ["f32", "bf16", "float32", "bfloat16"])
def test_dtype_string_form_accepted(tiny_zarr, dtype):
    root, _ = tiny_zarr
    d = _native.Damacy(**_base_kwargs(root, dtype=dtype))
    assert d is not None


def test_dtype_int_form_accepted(tiny_zarr):
    root, _ = tiny_zarr
    # DAMACY_BF16 = 1 (post-#16 enum: F32=0, BF16=1).
    d = _native.Damacy(**_base_kwargs(root, dtype=1))
    assert d is not None


def test_dtype_unknown_string_raises(tmp_path):
    kw = _base_kwargs(tmp_path, dtype="u16")  # post-#16: u16 is source-only
    with pytest.raises(ValueError, match="unknown dtype"):
        _native.Damacy(**kw)


# ---------- end-to-end push / pop / release ----------

def test_push_pop_release(tiny_zarr):
    root, uri = tiny_zarr
    d = _native.Damacy(**_base_kwargs(root))
    r = d.push([{"uri": uri, "aabb": [(0, 8), (0, 16)]}])
    assert r["status"] == "ok"
    assert r["consumed"] == 1

    b = d.pop()
    info = b.info
    # Leading N axis (batch_size=1) followed by zarr axes. dtype is the
    # *destination* (cfg.dtype); the u16 source casts to f32 in-kernel.
    assert info["shape"] == (1, 8, 16)
    assert info["dtype"] == "f32"
    assert info["batch_id"] == 0
    assert info["device_ptr"] != 0
    # release returns the slot to the pool; idempotent.
    b.release()
    b.release()


def test_unknown_uri_returns_notfound(tiny_zarr):
    root, _ = tiny_zarr
    d = _native.Damacy(**_base_kwargs(root))
    r = d.push([{"uri": "not_a_zarr", "aabb": [(0, 8), (0, 16)]}])
    assert r["status"] == "not found"
    assert r["consumed"] == 0


def test_unsupported_src_dtype_returns_dtype_error(tiny_zarr_no_cast):
    # The fixture is int64 — no cast path to f32 (post-#16). push must
    # surface DAMACY_DTYPE on the offending sample.
    root, uri = tiny_zarr_no_cast
    d = _native.Damacy(**_base_kwargs(root, dtype="f32"))
    r = d.push([{"uri": uri, "aabb": [(0, 8), (0, 16)]}])
    assert r["status"] == "dtype mismatch"
    assert r["consumed"] == 0


def test_oversize_chunk_surfaces_at_pop(tmp_path, write_zarr_script):
    # Mirrors test_damacy_caps.c::test_oversize_chunk_rejected: 8x16 u16
    # = 256 B inner chunk, runtime cap 128 B → planner rejects at pop.
    import shutil, subprocess
    if not shutil.which("uv"):
        pytest.skip("uv not on PATH")
    out = tmp_path / "foo"
    cmd = [
        "uv", "run", "--script", str(write_zarr_script),
        "--out", str(out), "--shape", "8,16", "--inner", "8,16",
        "--shard", "8,16", "--dtype", "uint16", "--codec", "blosc-lz4",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"write_zarr.py failed: {r.stderr}")

    kw = _base_kwargs(tmp_path)
    kw["max_chunk_uncompressed_bytes"] = 128  # 8x16 u16 = 256 B → over
    d = _native.Damacy(**kw)
    res = d.push([{"uri": "foo", "aabb": [(0, 8), (0, 16)]}])
    assert res["status"] == "ok"
    with pytest.raises(RuntimeError, match="invalid argument"):
        d.pop()


# ---------- stats ----------

def test_stats_shape_includes_gpu_bytes_committed(tiny_zarr):
    root, _ = tiny_zarr
    d = _native.Damacy(**_base_kwargs(root))
    s = d.stats()
    # New field surfaced from struct damacy_stats; lazy batch pool means
    # it's nonzero immediately after create (wave_init commits scratch).
    assert "gpu_bytes_committed" in s
    assert isinstance(s["gpu_bytes_committed"], int)
    assert s["gpu_bytes_committed"] > 0


def test_stats_gpu_bytes_grows_after_first_pop(tiny_zarr):
    root, uri = tiny_zarr
    d = _native.Damacy(**_base_kwargs(root))
    before = d.stats()["gpu_bytes_committed"]
    r = d.push([{"uri": uri, "aabb": [(0, 8), (0, 16)]}])
    assert r["status"] == "ok"
    b = d.pop()
    after = d.stats()["gpu_bytes_committed"]
    b.release()
    # The batch-output pool is lazily sized on the first pop, so the
    # committed count must strictly grow past the wave-init baseline.
    assert after > before


# ---------- log sink smoke ----------

def test_log_sink_routes_to_python_logger(caplog):
    # Drives a synthetic record from a freshly-spawned C thread through
    # the lock-free ring → drain thread → logging.getLogger("damacy").
    # caplog captures at WARNING by default; we use ERROR to be safe.
    with caplog.at_level(logging.ERROR, logger="damacy"):
        _native._log_emit_from_thread(_native.LOG_ERROR, "pytest-smoke")
        # Drain runs in a daemon thread; give it a beat.
        import time
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if any("pytest-smoke" in r.getMessage() for r in caplog.records):
                break
            time.sleep(0.02)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("pytest-smoke" in m for m in msgs), msgs
