"""Pydantic models shared by run.py and report.py.

Mirrors the scenario.json schema and the results.json emitted by
damacy_bench. Both scripts add this directory to sys.path before
importing.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

SrcDType = Literal["u8", "u16", "i16", "u32", "i32", "f16", "f32"]
DstDType = Literal["f32", "bf16"]
Codec = Literal["none", "zstd", "blosc-zstd"]

NUMPY_DTYPE = {
    "u8": "uint8",
    "u16": "uint16",
    "i16": "int16",
    "u32": "uint32",
    "i32": "int32",
    "f16": "float16",
    "f32": "float32",
}


class Dataset(BaseModel):
    store_root: str
    n_zarrs: int = Field(gt=0)
    # Explicit per-array URIs relative to store_root. When set, the bench
    # samples these real arrays (reading each one's shape from its
    # zarr.json) and the synthetic-generation fields below are unused.
    # When None, the synthetic path applies: arrays are named uri_fmt % i
    # and generated to a single uniform zarr_shape.
    uris: list[str] | None = None
    uri_fmt: str | None = None
    array_path: str = "scale0/image"
    zarr_shape: list[int] | None = None
    chunk_shape: list[int] | None = None
    shard_shape: list[int] | None = None
    # Source dtype assigned per-zarr cycling through this list. Length
    # must be >= 1 and ideally divide n_zarrs.
    dtypes: list[SrcDType] | None = None
    # Codec assigned per-zarr cycling through this list. Length must
    # divide n_zarrs (or be 1). `["zstd"]` reproduces the original
    # single-codec scenario.
    codecs: list[Codec] = Field(default_factory=lambda: ["zstd"])
    clevel: int = 3
    entropy: float = 0.5
    seed: int = 42

    @field_validator("zarr_shape", "chunk_shape", "shard_shape")
    @classmethod
    def _positive(cls, v: list[int] | None) -> list[int] | None:
        if v is not None and (not v or any(x <= 0 for x in v)):
            raise ValueError("shape entries must be positive")
        return v

    @model_validator(mode="after")
    def _gen_fields_present(self) -> "Dataset":
        if self.uris is None:
            missing = [
                f
                for f in ("uri_fmt", "zarr_shape", "chunk_shape", "shard_shape", "dtypes")
                if getattr(self, f) is None
            ]
            if missing:
                raise ValueError(
                    f"synthetic dataset (no `uris`) requires: {', '.join(missing)}"
                )
        elif not self.uris:
            raise ValueError("`uris` must be non-empty when present")
        return self

    def codec_for(self, zarr_idx: int) -> Codec:
        return self.codecs[zarr_idx % len(self.codecs)]

    def dtype_for(self, zarr_idx: int) -> SrcDType:
        assert self.dtypes is not None
        return self.dtypes[zarr_idx % len(self.dtypes)]


class Sampling(BaseModel):
    sample_shape: list[int]
    n_batches: int = Field(gt=0)
    n_warmup_batches: int = 0
    samples_per_batch: int = Field(gt=0)
    seed: int = 1234


class Pipeline(BaseModel):
    # Destination dtype on the assembled batch. Sources cast to this.
    dtype: DstDType = "f32"
    lookahead_samples: int = Field(gt=0)
    n_io_threads: int = Field(gt=0)
    metadata_io_concurrency: int = Field(default=32, gt=0)
    max_gpu_memory_mb: int = 0  # 0 → library default
    max_chunk_uncompressed_mb: int = 0  # 0 → library default
    # cap on coalesced read_op size; unset → library default, 0 is a real value
    max_read_op_kb: int | None = None
    n_array_meta_cache: int = 4096
    n_shard_index_cache: int = 16384
    n_chunk_layout_cache: int = 4096
    # Declared upper bound on shards a sample's AABB may intersect. Sizes the
    # n_shard_index_cache floor (>= lookahead_samples * max_shards_per_sample)
    # and caps per-sample shard enumeration at runtime.
    max_shards_per_sample: int = Field(default=64, gt=0)
    # Bench bypass: skip decode by flipping chunks to fill at parse +
    # assemble time. IO and input transfer still run; assemble broadcasts the
    # array's fill_value. Useful for isolating decode cost.
    bypass_decode: bool = False


class Consumer(BaseModel):
    hold_ms: float = 0.0


class LatencyModel(BaseModel):
    baseline_ns: int = Field(default=0, ge=0)
    lognormal_mu_ln_ns: float = 0.0
    lognormal_sigma_ln_ns: float = Field(default=0.0, ge=0.0)
    cap_ns: int = Field(default=0, ge=0)
    seed: int = Field(default=0, ge=0)


class Scenario(BaseModel):
    name: str = "scenario"
    dataset: Dataset
    sampling: Sampling
    pipeline: Pipeline
    consumer: Consumer = Field(default_factory=Consumer)
    metadata_latency: LatencyModel = Field(default_factory=LatencyModel)


# --- results -----------------------------------------------------------------


class Stage(BaseModel):
    name: str
    unit: str
    count: int
    ms_total: float
    ms_avg: float
    ms_best: float
    input_bytes: float
    output_bytes: float


class Timings(BaseModel):
    init: float
    time_to_first_batch: float
    wall: float
    consumer_block: float = 0.0
    consumer_push: float = 0.0
    consumer_pop_wait: float = 0.0


class MetadataOpLatency(BaseModel):
    """Real, measured submit->completion latency for one io_uring op kind.

    Buckets are a log2-scale histogram on nanoseconds: bucket i counts ops whose
    latency satisfies floor(log2(ns)) == i. Percentiles are derived here from the
    raw buckets so the estimator can evolve without touching the C ABI."""

    op: str
    count: int = 0
    sum_ns: int = 0
    max_ns: int = 0
    buckets: list[int] = Field(default_factory=list)

    def avg_ns(self) -> float:
        return self.sum_ns / self.count if self.count else 0.0

    def percentile_ns(self, q: float) -> float:
        """Geometric-mean estimate of the q-quantile (0..1) from the buckets.

        Bucket i covers [2^i, 2^(i+1)); we return the bucket's geometric mean
        2^(i+0.5) as the estimate, clamped to max_ns for the tail."""
        if self.count == 0:
            return 0.0
        rank = q * self.count
        cum = 0
        for i, n in enumerate(self.buckets):
            cum += n
            if cum >= rank:
                est = 2.0 ** (i + 0.5) if i > 0 else 0.0
                return min(est, float(self.max_ns)) if self.max_ns else est
        return float(self.max_ns)


class Counters(BaseModel):
    samples_pushed: int
    batches_emitted: int
    waves_emitted: int
    chunks_dispatched: int
    chunks_planned: int = 0
    chunks_to_load: int = 0
    reads_issued: int = 0
    distinct_zarrs: int
    distinct_shards: int
    array_meta_hits: int
    array_meta_misses: int
    shard_index_hits: int
    shard_index_misses: int
    chunk_layout_hits: int
    chunk_layout_misses: int
    metadata_latency_ops: int = 0
    metadata_latency_stat_ops: int = 0
    metadata_latency_submit_ops: int = 0
    metadata_latency_active: int = 0
    metadata_latency_max_active: int = 0
    metadata_latency_total_sleep_ns: int = 0
    metadata_latency_max_sleep_ns: int = 0
    metadata_backend_read_jobs: int = 0
    metadata_backend_read_active: int = 0
    metadata_backend_read_max_active: int = 0
    metadata_op_latency: list[MetadataOpLatency] = Field(default_factory=list)
    gpu_bytes_committed: int = 0


class Derived(BaseModel):
    bytes_per_sample: int
    throughput_mb_s: float
    stage_concurrency: float
    chunks_per_batch: float
    chunks_per_wave: float


class Results(BaseModel):
    scenario: Scenario
    timings_ms: Timings
    stages: list[Stage]
    counters: Counters
    derived: Derived


# --- helpers -----------------------------------------------------------------


def zarr_subdir_fmt(uri_fmt: str, array_path: str) -> str:
    """Strip the array sub-path from the URI template to get the per-zarr
    on-disk directory template. Empty string when the array is at the
    store root (single-zarr scenarios)."""
    if uri_fmt == array_path:
        return ""
    suffix = "/" + array_path
    if not uri_fmt.endswith(suffix):
        raise ValueError(
            f"uri_fmt {uri_fmt!r} must end with /{array_path!r}; "
            "set dataset.array_path to override"
        )
    return uri_fmt[: -len(suffix)]


def format_subdir(sub_fmt: str, i: int) -> str:
    return (sub_fmt % i) if "%" in sub_fmt else sub_fmt
