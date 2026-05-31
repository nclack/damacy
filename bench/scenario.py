"""Pydantic models shared by run.py and report.py.

Mirrors the scenario.json schema and the results.json emitted by
damacy_bench. Both scripts add this directory to sys.path before
importing.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

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
    uri_fmt: str
    array_path: str = "scale0/image"
    zarr_shape: list[int]
    chunk_shape: list[int]
    shard_shape: list[int]
    # Source dtype assigned per-zarr cycling through this list. Length
    # must be >= 1 and ideally divide n_zarrs.
    dtypes: list[SrcDType]
    # Codec assigned per-zarr cycling through this list. Length must
    # divide n_zarrs (or be 1). `["zstd"]` reproduces the original
    # single-codec scenario.
    codecs: list[Codec] = Field(default_factory=lambda: ["zstd"])
    clevel: int = 3
    entropy: float = 0.5
    seed: int = 42

    @field_validator("zarr_shape", "chunk_shape", "shard_shape")
    @classmethod
    def _positive(cls, v: list[int]) -> list[int]:
        if not v or any(x <= 0 for x in v):
            raise ValueError("shape entries must be positive")
        return v

    @field_validator("codecs", "dtypes")
    @classmethod
    def _nonempty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("must be non-empty")
        return v

    def codec_for(self, zarr_idx: int) -> Codec:
        return self.codecs[zarr_idx % len(self.codecs)]

    def dtype_for(self, zarr_idx: int) -> SrcDType:
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
    n_prefetch_io_threads: int = Field(default=16, ge=0)
    max_gpu_memory_mb: int = 0  # 0 → library default
    max_chunk_uncompressed_mb: int = 0  # 0 → library default
    max_read_op_kb: int = 0  # cap on coalesced read_op size; 0 → library default
    n_array_meta_cache: int = 4096
    n_shard_index_cache: int = 16384
    n_chunk_layout_cache: int = 4096
    # Bench bypass: skip decode by flipping chunks to fill at parse +
    # assemble time. IO and input transfer still run; assemble broadcasts the
    # array's fill_value. Useful for isolating decode cost.
    bypass_decode: bool = False


class Consumer(BaseModel):
    hold_ms: float = 0.0


class Scenario(BaseModel):
    name: str = "scenario"
    dataset: Dataset
    sampling: Sampling
    pipeline: Pipeline
    consumer: Consumer = Field(default_factory=Consumer)


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


class Counters(BaseModel):
    samples_pushed: int
    batches_emitted: int
    batches_truncated: int
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
