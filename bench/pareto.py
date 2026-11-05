#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Plan, run, and publish retained reader comparisons."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3
DTYPES = {"uint8": ("u8", 1), "uint16": ("u16", 2), "float32": ("f32", 4)}
METRICS_VERSION = 3
NEW_METRICS = {"peak_active_reads"}
IO_METRICS = {"read_iops", "read_gib_s", "mean_read_bytes", "reads_per_useful_gib"}
IO_METRICS.update(
    {
        "device_iops",
        "device_read_gib_s",
        "device_mean_read_bytes",
        "device_read_amplification",
        "idle_device_iops",
        "idle_device_gib_s",
        "cache_preparation_seconds",
    }
)


def read_json(path):
    def invalid(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    return json.loads(Path(path).read_text(), parse_constant=invalid)


def encoded(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def identity(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Refusing to replace different data: {path}")
        return
    with path.open("x") as stream:
        stream.write(data)


def positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def relative_path(root, value):
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in value:
        raise ValueError(f"Expected a relative file path: {value}")
    return Path(root).joinpath(*path.parts)


def corpus_inputs(document, names):
    expected = {
        "name": "chucky-image-corpus",
        "version": 2,
        "encoding": "raw",
        "byte_order": "little",
        "axes": ["plane", "y", "x"],
        "order": "C",
    }
    if document.get("format") != expected:
        raise ValueError("Expected a format-2, little-endian microscopy image corpus")
    inputs = {}
    for dataset in document["datasets"]:
        for asset in dataset["assets"]:
            if asset["id"] not in names:
                continue
            if asset["id"] in inputs:
                raise ValueError(f"Duplicate input: {asset['id']}")
            shape = asset["shape"]
            if len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape):
                raise ValueError("Inputs must have positive plane, Y, and X dimensions")
            if asset["dtype"] not in DTYPES:
                raise ValueError(f"Unsupported input type: {asset['dtype']}")
            digest = asset["sha256"]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Each input needs a SHA256 checksum")
            relative_path(Path("."), asset["path"])
            inputs[asset["id"]] = {
                **{
                    key: asset[key]
                    for key in ("id", "name", "path", "dtype", "shape", "sha256")
                },
                "source": dataset["source"],
                "dataset_id": dataset["id"],
                "dataset_version": dataset["version"],
                "modality": dataset["modality"],
            }
    if set(inputs) != set(names):
        raise ValueError(f"Missing inputs: {sorted(set(names) - set(inputs))}")
    return [inputs[name] for name in names]


def chunk_shape(byte_count, itemsize):
    positive_int(byte_count, "chunk_bytes")
    elements, remainder = divmod(byte_count, itemsize)
    if remainder or elements <= 0 or elements & (elements - 1):
        raise ValueError("Chunk sizes must contain a power-of-two number of pixels")
    bits = elements.bit_length() - 1
    return [1, 2 ** (bits // 2), 2 ** (bits - bits // 2)]


def make_plan(definition, corpus, *, pilot=False):
    if definition["version"] == 2:
        from pareto_mixture import make_mixture_plan

        plan = make_mixture_plan(definition, corpus, pilot=pilot)
        schedule(plan)
        return plan
    if definition["version"] != 1:
        raise ValueError("Unsupported study definition version")
    if corpus["version"] != definition["corpus_version"]:
        raise ValueError("Corpus version does not match the study definition")
    positive_int(definition["query_seed"], "query_seed")
    names = definition["pilot_inputs"] if pilot else definition["inputs"]
    if len(names) != len(set(names)) or not names:
        raise ValueError("Input IDs must be unique and nonempty")
    inputs = corpus_inputs(corpus, names)
    profile = definition["pilot_profile"] if pilot else definition["profile"]
    cache_policy = definition.get("cache_policy", "preloaded files")
    if cache_policy not in {"cold", "preloaded files"}:
        raise ValueError("Cache policy must be cold or preloaded files")
    for key in ("repeats", "bytes_per_pass", "warmup_batches"):
        positive_int(profile[key], key)
    for key in ("min_seconds", "reference_seconds", "timeout_seconds"):
        positive_number(profile[key], key)
    if profile["min_seconds"] > 600 or profile["reference_seconds"] > 600:
        raise ValueError("Measurement windows must be at most 600 seconds")
    if profile["timeout_seconds"] <= max(
        profile["min_seconds"], profile["reference_seconds"]
    ):
        raise ValueError("Process timeout must exceed the measurement window")
    if set(definition["backends"]) - {"cpu", "cuda"} or not definition["backends"]:
        raise ValueError("Backends must be cpu or cuda")
    if len(set(definition["backends"])) != len(definition["backends"]):
        raise ValueError("Duplicate backend")
    variants, cases = {}, []
    for asset in inputs:
        _, itemsize = DTYPES[asset["dtype"]]
        for size in definition["chunk_bytes"]:
            chunks = chunk_shape(size, itemsize)
            shards = definition["shard_shape"]
            if len(shards) != 3 or any(
                type(s) is not int or s <= 0 or s % c
                for s, c in zip(shards, chunks, strict=True)
            ):
                raise ValueError(
                    "Fixed shard dimensions must be multiples of every chunk shape"
                )
            for codec in definition["codecs"]:
                if codec not in {"none", "zstd", "blosc-zstd"}:
                    raise ValueError(f"Unsupported reader codec: {codec}")
                blocks = [None]
                if codec == "blosc-zstd":
                    blocks = sorted(
                        {
                            size if b == "chunk" else positive_int(b, "block_bytes")
                            for b in definition["blosc_block_bytes"]
                        }
                    )
                    blocks = [b for b in blocks if b <= size]
                for block in blocks:
                    setting = {
                        "codec": codec,
                        "chunk_bytes": size,
                        "block_bytes": block,
                    }
                    if pilot and setting not in definition["pilot_settings"]:
                        continue
                    if block and math.ceil(size / block) > definition["pipeline"].get(
                        "max_substreams_per_chunk", 32
                    ):
                        raise ValueError("Blosc blocks exceed max_substreams_per_chunk")
                    variant = {
                        **setting,
                        "input_id": asset["id"],
                        "input_sha256": asset["sha256"],
                        "dtype": asset["dtype"],
                        "shape": asset["shape"],
                        "chunk_shape": chunks,
                        "shard_shape": shards,
                        "level": 0 if codec == "none" else definition["level"],
                        "shuffle": "bitshuffle" if codec == "blosc-zstd" else "none",
                    }
                    variant_id = identity(variant)[:20]
                    variants[variant_id] = {"id": variant_id, **variant}
                    for pattern in definition["patterns"]:
                        if pattern["pattern"] == "planes":
                            sample = [1, *asset["shape"][1:]]
                        elif pattern["pattern"] == "random":
                            sample = pattern["sample_shape"]
                        else:
                            raise ValueError("Read patterns must be random or planes")
                        if len(sample) != 3 or any(
                            type(s) is not int or s <= 0 or s > n
                            for s, n in zip(sample, asset["shape"], strict=True)
                        ):
                            raise ValueError(f"Sample does not fit {asset['id']}")
                        batch_size = positive_int(
                            pattern["samples_per_batch"], "samples_per_batch"
                        )
                        batches = math.ceil(
                            profile["bytes_per_pass"]
                            / (math.prod(sample) * itemsize * batch_size)
                        )
                        if pattern["pattern"] == "planes":
                            cycle = asset["shape"][0] // math.gcd(
                                asset["shape"][0], batch_size
                            )
                            batches = math.ceil(batches / cycle) * cycle
                        if cache_policy == "cold":
                            batches = 1
                            if pattern["pattern"] == "planes":
                                batch_size = min(batch_size, asset["shape"][0])
                        for backend in definition["backends"]:
                            case = {
                                "variant_id": variant_id,
                                "input_id": asset["id"],
                                "backend": backend,
                                "workload": pattern["id"],
                                "sampling": {
                                    "pattern": pattern["pattern"],
                                    "sample_shape": sample,
                                    "samples_per_batch": batch_size,
                                    "n_batches": batches,
                                    "n_warmup_batches": profile["warmup_batches"],
                                    "min_seconds": profile["min_seconds"],
                                    "seed": definition["query_seed"],
                                },
                            }
                            cases.append({"id": identity(case)[:20], **case})
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("The study must contain distinct measurement cases")
    plan = {
        "version": 1,
        "id": definition["id"],
        "label": definition["label"],
        "pilot": pilot,
        "definition_sha256": identity(definition),
        "corpus_id": corpus["id"],
        "corpus_version": corpus["version"],
        "corpus_sha256": identity(corpus),
        "inputs": inputs,
        "variants": list(variants.values()),
        "cases": cases,
        "profile": profile,
        "pipeline": definition["pipeline"],
        "order_seed": definition["order_seed"],
        "reference_setting": definition["reference_setting"],
        "cache_policy": cache_policy,
    }
    schedule(plan)
    return plan


def case_variant(case, variants):
    if "variant_ids" not in case:
        return variants[case["variant_id"]]
    selected = [variants[key] for key in case["variant_ids"]]
    first = selected[0]
    setting = {
        key: first[key]
        for key in ("codec", "chunk_bytes", "block_bytes", "level", "shuffle")
    }
    if any(
        any(value[key] != expected for key, expected in setting.items())
        for value in selected
    ):
        raise ValueError(
            "A mixture must use the same codec and byte settings for every source"
        )
    return {
        **setting,
        "id": case["variant_id"],
        "input_id": case["input_id"],
        "dtype": "mixed",
        "chunk_shape": [],
        "shard_shape": [],
        "input_sha256": identity([value["input_sha256"] for value in selected]),
        "logical_bytes": sum(
            math.prod(value["shape"]) * DTYPES[value["dtype"]][1] for value in selected
        ),
        "stored_bytes": sum(value.get("stored_bytes", 0) for value in selected),
        "actual_block_bytes": sorted(
            {size for value in selected for size in value.get("actual_block_bytes", [])}
        ),
        "files": [item for value in selected for item in value.get("files", [])],
        "sources": [
            {key: item for key, item in value.items() if key != "files"}
            for value in selected
        ],
    }


def is_cold(plan):
    return plan.get("cache_policy") in {"cold", "cold-pass"}


def schedule(plan, backend=None):
    variants = {v["id"]: v for v in plan["variants"]}
    groups = {}
    for case in plan["cases"]:
        if backend is None or case["backend"] == backend:
            key = (case["input_id"], case["backend"], case["workload"])
            groups.setdefault(key, []).append(case)
    if not groups:
        raise ValueError("No cases match the selected backend")
    tasks = []
    for repetition in range(plan["profile"]["repeats"]):
        rng = random.Random(plan["order_seed"] + repetition)
        order = list(groups)
        rng.shuffle(order)
        for key in order:
            cases = list(groups[key])
            rng.shuffle(cases)
            reference = [
                c
                for c in cases
                if all(
                    case_variant(c, variants)[k] == v
                    for k, v in plan["reference_setting"].items()
                )
            ]
            if len(reference) != 1:
                raise ValueError(
                    f"Each condition needs exactly one reference setting: {key}"
                )
            for role, case in [
                ("reference-before", reference[0]),
                *[("sample", c) for c in cases],
                ("reference-after", reference[0]),
            ]:
                task = {"round": repetition + 1, "role": role, "case_id": case["id"]}
                tasks.append({"id": identity(task)[:20], **task})
    return tasks


def require_compute():
    if shutil.which("srun") and not os.environ.get("SLURM_JOB_ID"):
        raise ValueError(
            "Preparation and measurement require a compute allocation on this cluster"
        )


def command_output(args):
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def source_record():
    frozen = ROOT / ".pareto-source.json"
    if frozen.exists():
        record = read_json(frozen)
        files = {name: file_hash(relative_path(ROOT, name)) for name in record["files"]}
        if files != record["files"]:
            raise ValueError("Frozen benchmark sources changed")
        return {**record, "sha256": identity(files)}
    tracked = command_output(
        [
            "git",
            "-C",
            str(ROOT),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "src",
            "bench",
            "cmake",
            "CMakeLists.txt",
        ]
    )
    names = set((tracked or "").splitlines()) | {
        "bench/pareto.py",
        "bench/pareto_data.py",
    }
    files = {
        name: file_hash(ROOT / name)
        for name in sorted(names)
        if (ROOT / name).is_file()
    }
    return {
        "revision": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "local_changes": bool(
            command_output(
                [
                    "git",
                    "-C",
                    str(ROOT),
                    "status",
                    "--porcelain",
                    "--",
                    "src",
                    "bench",
                    "cmake",
                    "CMakeLists.txt",
                ]
            )
        ),
        "sha256": identity(files),
        "files": files,
    }


def machine_record(data):
    cpu = command_output(["lscpu", "--json"])
    cpu_fields = {
        "Architecture:",
        "Model name:",
        "Vendor ID:",
        "Socket(s):",
        "Core(s) per socket:",
        "Thread(s) per core:",
        "NUMA node(s):",
    }
    cpu_record = (
        {
            row["field"].rstrip(":"): row["data"]
            for row in json.loads(cpu)["lscpu"]
            if row["field"] in cpu_fields
        }
        if cpu
        else None
    )
    mount = command_output(
        ["findmnt", "--json", "--target", str(data), "--output", "FSTYPE"]
    )
    return {
        "host": platform.node(),
        "system": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "cpu": cpu_record,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "filesystem": json.loads(mount) if mount else None,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
    }


def local_nvme_record(data):
    mount = command_output(
        [
            "findmnt",
            "--json",
            "--target",
            str(data),
            "--output",
            "FSTYPE,SOURCE,MAJ:MIN",
        ]
    )
    if not mount:
        raise ValueError("Cannot identify the data filesystem")
    filesystem = json.loads(mount)["filesystems"][0]
    if filesystem["fstype"] not in {"xfs", "ext4"}:
        raise ValueError("Cold study requires a verified local XFS or ext4 filesystem")
    dev = Path(data).stat().st_dev
    number = f"{os.major(dev)}:{os.minor(dev)}"
    if filesystem["maj:min"] != number:
        raise ValueError("Data filesystem and mount device disagree")
    root = Path("/sys/dev/block") / number
    if not root.exists():
        raise ValueError("The data block device is not visible in sysfs")
    devices = block_devices(root)
    if not devices:
        raise ValueError("No local NVMe devices found for the data filesystem")
    return {"filesystem": filesystem, "devices": devices}


def block_devices(root, visited=None):
    visited = set() if visited is None else visited
    root = root.resolve(strict=True)
    if root in visited:
        return []
    visited.add(root)
    if (root / "partition").exists():
        return block_devices(root.parent, visited)
    slaves = sorted((root / "slaves").iterdir()) if (root / "slaves").exists() else []
    if slaves:
        return [device for slave in slaves for device in block_devices(slave, visited)]
    transport = next(
        (
            path.read_text().strip()
            for parent in (root, *root.parents)
            if (path := parent / "transport").is_file()
        ),
        None,
    )
    if not root.name.startswith("nvme") or transport != "pcie":
        raise ValueError(f"Data device {root.name} is not verified PCIe NVMe")
    model = next(
        (
            path.read_text().strip()
            for parent in (root, *root.parents)
            if (path := parent / "model").is_file()
        ),
        None,
    )
    if not model or not (root / "stat").is_file():
        raise ValueError("NVMe identity or device counters are unavailable")
    return [
        {
            "name": root.name,
            "model": model,
            "transport": transport,
            "bytes": int((root / "size").read_text()) * 512,
            "stat_path": str(root / "stat"),
        }
    ]


def prepare_cache(files):
    count = 0
    for path in files:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                count += len(block)
    return {"policy": "preloaded files", "files": len(files), "bytes": count}


def scenario_for(plan, case, data, *, reference=False, variant=None, storage=None):
    sampling = copy.deepcopy(case["sampling"])
    if reference:
        sampling["min_seconds"] = plan["profile"]["reference_seconds"]
    pipeline = {**plan["pipeline"], "executor": case["backend"]}
    if pipeline["lookahead_samples"] < sampling["samples_per_batch"]:
        raise ValueError("Lookahead must hold at least one batch")
    scenario = {
        "name": case["id"],
        "dataset": {
            "store_root": str(Path(data).resolve()),
            "n_zarrs": len(case.get("variant_ids", [case["variant_id"]])),
            "uris": [
                f"{key}/array" for key in case.get("variant_ids", [case["variant_id"]])
            ],
        },
        "sampling": sampling,
        "pipeline": pipeline,
        "consumer": {"hold_ms": 0},
    }
    if is_cold(plan):
        if not variant or not storage:
            raise ValueError("Cold scenarios require files and verified NVMe devices")
        scenario["cache"] = {
            "policy": plan["cache_policy"],
            "files": [
                str(relative_path(Path(data).resolve(), item["path"]))
                for item in variant["files"]
                if not item["path"].endswith("/zarr.json")
            ],
            "device_stats": [device["stat_path"] for device in storage["devices"]],
        }
    if "query_id" in sampling:
        sampling["query_file"] = str(
            Path(data).resolve() / "queries" / f"{sampling['query_id']}.bin"
        )
    return scenario


def observation_metrics(result, case, variant):
    passes = positive_int(
        result.get("measurement_passes"), "measurement_passes (rebuild damacy_bench)"
    )
    sampling = result["scenario"]["sampling"]
    for key in ("pattern", "sample_shape", "n_batches", "samples_per_batch", "seed"):
        if sampling[key] != case["sampling"][key]:
            raise ValueError("Measured queries disagree with the planned case")
    batches = passes * sampling["n_batches"]
    if result["counters"]["batches_emitted"] != batches:
        raise ValueError("Measured batch count disagrees with completed query passes")
    wall_s = positive_number(result["timings_ms"]["wall"], "wall time") / 1000
    if wall_s + 0.001 < sampling["min_seconds"]:
        raise ValueError("Measurement did not reach its requested window")
    input_bytes = (
        batches
        * sampling["samples_per_batch"]
        * math.prod(sampling["sample_shape"])
        * (1 if "query_summary" in case else DTYPES[variant["dtype"]][1])
    )
    if "query_summary" in case:
        summary = case["query_summary"]
        expected = {
            "samples_per_pass": summary["samples"],
            "source_bytes_per_pass": summary["useful_bytes"],
            "samples_per_source": [
                item["samples"] for item in summary["source_samples"]
            ],
        }
        if result.get("query_stream") != expected:
            raise ValueError(
                "Native query counts or source byte sizes disagree with the plan"
            )
        if summary["samples"] != sampling["n_batches"] * sampling["samples_per_batch"]:
            raise ValueError("Query stream does not contain complete batches")
        input_bytes = passes * summary["useful_bytes"]
    returned_bytes = (
        batches * sampling["samples_per_batch"] * result["derived"]["bytes_per_sample"]
    )
    stages = {stage["name"]: stage for stage in result["stages"]}
    read_bytes = stages["io"]["input_bytes"]
    decoded_bytes = stages["decode"]["output_bytes"]
    for name, value in (("read bytes", read_bytes), ("decoded bytes", decoded_bytes)):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {name}")
    if read_bytes <= 0:
        raise ValueError("No encoded file reads were measured")
    reads = result["counters"].get("reads_issued")
    if reads is not None:
        positive_int(reads, "reads_issued")
    peak_reads = result["counters"].get("data_backend_read_max_active")
    if peak_reads is not None:
        positive_int(peak_reads, "peak active reads")
        if peak_reads > result["scenario"]["pipeline"]["n_io_threads"]:
            raise ValueError("Peak active reads exceed file workers")
    elif "query_summary" in case:
        raise ValueError("Mixed studies require measured read concurrency")
    cold = result.get("cold_reads")
    policy = result["scenario"].get("cache", {}).get("policy")
    if policy in {"cold", "cold-pass"}:
        if (
            not cold
            or cold["passes"] != passes
            or (policy == "cold" and sampling["n_batches"] != 1)
        ):
            raise ValueError("Cold reads require a checked cold start for every pass")
        if cold["resident_pages_after"] != 0:
            raise ValueError("Cold reads still had resident file pages")
        for key in ("files", "pages_per_check", "device_reads", "device_read_bytes"):
            positive_int(cold[key], key)
        if cold["device_reads"] < passes:
            raise ValueError("Cold batches did not issue device reads")
        positive_number(cold["idle_seconds"], "idle seconds")
        positive_number(cold["preparation_ms"], "cache preparation time")
        for key in (
            "idle_device_reads",
            "idle_device_read_bytes",
            "idle_device_writes",
            "device_writes",
        ):
            if type(cold[key]) is not int or cold[key] < 0:
                raise ValueError(f"Invalid {key}")
    elif cold is not None:
        raise ValueError("Cold read counters were recorded for a cached scenario")
    return {
        "peak_active_reads": peak_reads,
        "useful_gib_s": input_bytes / GIB / wall_s,
        "returned_gib_s": returned_bytes / GIB / wall_s,
        "read_iops": reads / wall_s if reads is not None else None,
        "read_gib_s": read_bytes / GIB / wall_s,
        "mean_read_bytes": read_bytes / reads if reads is not None else None,
        "reads_per_useful_gib": reads * GIB / input_bytes
        if reads is not None
        else None,
        "device_iops": cold["device_reads"] / wall_s if cold else None,
        "device_read_gib_s": cold["device_read_bytes"] / GIB / wall_s if cold else None,
        "device_mean_read_bytes": cold["device_read_bytes"] / cold["device_reads"]
        if cold
        else None,
        "device_read_amplification": cold["device_read_bytes"] / input_bytes
        if cold
        else None,
        "idle_device_iops": cold["idle_device_reads"] / cold["idle_seconds"]
        if cold
        else None,
        "idle_device_gib_s": cold["idle_device_read_bytes"] / GIB / cold["idle_seconds"]
        if cold
        else None,
        "cache_preparation_seconds": cold["preparation_ms"] / 1000 if cold else None,
        "useful_bytes": input_bytes,
        "wall_seconds": wall_s,
        "read_amplification": read_bytes / input_bytes,
        "decode_amplification": decoded_bytes / input_bytes
        if decoded_bytes > 0
        else None,
        "first_batch_ms": result["timings_ms"]["time_to_first_batch"],
        "passes": passes,
        "batches": batches,
    }


def validate_saved_metrics(row, metrics):
    version = row.get("metrics_version", 1)
    if version == 1:
        expected = {
            key: value
            for key, value in metrics.items()
            if key not in IO_METRICS | NEW_METRICS
        }
    elif version == 2:
        expected = {
            key: value for key, value in metrics.items() if key not in NEW_METRICS
        }
    elif version == METRICS_VERSION:
        expected = metrics
    else:
        raise ValueError(f"Unsupported observation metrics version: {version}")
    if row["metrics"] != expected:
        raise ValueError("Observation metrics disagree with the native result")


def validate_prepared(plan, prepared):
    if prepared["plan_sha256"] != identity(plan):
        raise ValueError("Prepared files belong to a different study plan")
    variants = {variant["id"]: variant for variant in prepared["variants"]}
    if len(variants) != len(prepared["variants"]) or set(variants) != {
        v["id"] for v in plan["variants"]
    }:
        raise ValueError("Prepared layouts disagree with the plan")
    for expected in plan["variants"]:
        actual = variants[expected["id"]]
        if any(actual[key] != value for key, value in expected.items()):
            raise ValueError("Prepared layout settings changed")
        if (
            actual["logical_bytes"]
            != math.prod(expected["shape"]) * DTYPES[expected["dtype"]][1]
        ):
            raise ValueError("Prepared logical byte count changed")
        if not actual["files"] or len(
            {item["path"] for item in actual["files"]}
        ) != len(actual["files"]):
            raise ValueError("Prepared files are missing or duplicated")
        for item in actual["files"]:
            relative_path(Path("."), item["path"])
            if not item["path"].startswith(f"{expected['id']}/array/"):
                raise ValueError("Prepared file belongs to another layout")
            positive_int(item["bytes"], "stored file length")
        if actual["stored_bytes"] != sum(item["bytes"] for item in actual["files"]):
            raise ValueError("Stored byte count disagrees with the file inventory")


def run_study(args):
    require_compute()
    plan = read_json(args.plan)
    prepared = read_json(args.data / "prepared.json")
    validate_prepared(plan, prepared)
    from pareto_mixture import write_queries

    write_queries(plan, args.data)
    write_queries(plan, args.output)
    tasks = schedule(plan, args.backend)
    cases = {case["id"]: case for case in plan["cases"]}
    variants = {v["id"]: v for v in prepared["variants"]}
    bench = args.bench.resolve(strict=True)
    machine = machine_record(args.data)
    if is_cold(plan):
        machine["local_storage"] = local_nvme_record(args.data)
    if (
        any(cases[t["case_id"]]["backend"] == "cuda" for t in tasks)
        and not machine["gpu"]
    ):
        raise ValueError("The selected cases require a CUDA GPU")
    previous = (
        read_json(args.output / "context.json")
        if (args.output / "context.json").exists()
        else {}
    )
    context = {
        "session": previous.get("session", uuid.uuid4().hex),
        "started": previous.get("started", datetime.now(UTC).isoformat()),
        "plan_sha256": identity(plan),
        "prepared_sha256": identity(prepared),
        "binary_sha256": file_hash(bench),
        "binary_name": bench.name,
        "source": source_record(),
        "machine": machine,
        "storage": args.storage,
        "cache_policy": plan.get("cache_policy", "preloaded files"),
        "backend": args.backend,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "context.json", context)
    write_json(args.output / "plan.json", plan)
    write_json(args.output / "prepared.json", prepared)
    write_json(args.output / "schedule.json", tasks)
    for variant in variants.values():
        for item in variant["files"]:
            path = relative_path(args.data, item["path"])
            if is_cold(plan) and (
                not path.resolve().is_relative_to(args.data.resolve())
                or path.stat().st_dev != args.data.stat().st_dev
            ):
                raise ValueError(
                    "Cold files must stay on the verified local filesystem"
                )
            if (
                path.stat().st_size != item["bytes"]
                or file_hash(path) != item["sha256"]
            ):
                raise ValueError(f"Prepared file changed: {path}")
    for number, task in enumerate(tasks, 1):
        destination = args.output / "observations" / f"{task['id']}.json"
        if destination.exists():
            saved = read_json(destination)
            if saved["task"] != task or saved["context_sha256"] != identity(context):
                raise ValueError(
                    "Cannot resume observations from another measurement condition"
                )
            continue
        case = cases[task["case_id"]]
        variant = case_variant(case, variants)
        scenario = scenario_for(
            plan,
            case,
            args.data,
            reference=task["role"] != "sample",
            variant=variant,
            storage=machine.get("local_storage"),
        )
        query_path = (
            Path(scenario["sampling"]["query_file"])
            if "query_file" in scenario["sampling"]
            else None
        )
        if query_path and file_hash(query_path) != case["sampling"]["query_sha256"]:
            raise ValueError("Query stream changed before measurement")
        task_dir = args.output / "processes" / task["id"]
        task_dir.mkdir(parents=True, exist_ok=True)
        scenario_path = task_dir / "scenario.json"
        write_json(scenario_path, scenario)
        cache = (
            {"policy": plan["cache_policy"]}
            if is_cold(plan)
            else prepare_cache(
                [relative_path(args.data, item["path"]) for item in variant["files"]]
            )
        )
        started = datetime.now(UTC).isoformat()
        print(
            f"[{number}/{len(tasks)}] {case['input_id']} {case['backend']} {case['workload']} "
            f"{variant['codec']} {variant['chunk_bytes']} / {variant['block_bytes']} {task['role']}",
            flush=True,
        )
        before = time.monotonic()
        with (
            (task_dir / "stdout.json").open("w") as stdout,
            (task_dir / "stderr.txt").open("w") as stderr,
        ):
            process = subprocess.run(
                [str(bench), str(scenario_path.resolve())],
                stdout=stdout,
                stderr=stderr,
                timeout=plan["profile"]["timeout_seconds"],
                check=False,
            )
        elapsed = time.monotonic() - before
        if query_path and file_hash(query_path) != case["sampling"]["query_sha256"]:
            raise ValueError("Query stream changed during measurement")
        if process.returncode:
            raise ValueError(
                f"Benchmark failed ({process.returncode}); see {task_dir / 'stderr.txt'}"
            )
        result = read_json(task_dir / "stdout.json")
        if result["scenario"] != scenario:
            raise ValueError("Benchmark echoed a different scenario")
        metrics = observation_metrics(result, case, variant)
        if metrics["wall_seconds"] > elapsed * 1.01:
            raise ValueError("Measured wall time exceeds the supervising process clock")
        write_json(
            destination,
            {
                "task": task,
                "context_sha256": identity(context),
                "started": started,
                "process_seconds": elapsed,
                "cache": cache,
                "metrics_version": METRICS_VERSION,
                "metrics": metrics,
                "result": result,
            },
        )
    write_json(
        args.output / "complete.json",
        {"tasks": len(tasks), "context_sha256": identity(context)},
    )
    print(f"Completed {len(tasks)} executions: {args.output}")


def spread(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def summarize(directory):
    context = read_json(directory / "context.json")
    plan = read_json(directory / "plan.json")
    prepared = read_json(directory / "prepared.json")
    tasks = read_json(directory / "schedule.json")
    if read_json(directory / "complete.json") != {
        "tasks": len(tasks),
        "context_sha256": identity(context),
    }:
        raise ValueError("Study is incomplete or its context has changed")
    if (
        identity(plan) != context["plan_sha256"]
        or identity(prepared) != context["prepared_sha256"]
    ):
        raise ValueError("Study plan or prepared-file record changed")
    validate_prepared(plan, prepared)
    for query in plan.get("query_sets", []):
        if (
            file_hash(directory / "queries" / f"{query['id']}.bin")
            != query["summary"]["sha256"]
        ):
            raise ValueError("Retained query stream changed")
    if tasks != schedule(plan, context["backend"]):
        raise ValueError("Measurement schedule does not match the plan")
    cases = {case["id"]: case for case in plan["cases"]}
    variants = {v["id"]: v for v in prepared["variants"]}
    samples, references = {}, {}
    observations = []
    for task in tasks:
        row = read_json(directory / "observations" / f"{task['id']}.json")
        if row["task"] != task or row["context_sha256"] != identity(context):
            raise ValueError("Observation identity mismatch")
        case = cases[task["case_id"]]
        expected_scenario = scenario_for(
            plan,
            case,
            row["result"]["scenario"]["dataset"]["store_root"],
            reference=task["role"] != "sample",
            variant=case_variant(case, variants),
            storage=context["machine"].get("local_storage"),
        )
        if row["result"]["scenario"] != expected_scenario:
            raise ValueError(
                "Observation scenario disagrees with the measurement schedule"
            )
        metrics = observation_metrics(row["result"], case, case_variant(case, variants))
        validate_saved_metrics(row, metrics)
        row = {**row, "metrics_version": METRICS_VERSION, "metrics": metrics}
        condition = (case["input_id"], case["backend"], case["workload"])
        if task["role"] == "sample":
            samples.setdefault(case["id"], []).append(row)
        else:
            references.setdefault(condition, []).append(metrics["useful_gib_s"])
        public = copy.deepcopy(row)
        public["result"]["scenario"]["dataset"] = {
            "input_id": case["input_id"],
            "variant_id": case["variant_id"],
        }
        if "cache" in public["result"]["scenario"]:
            public["result"]["scenario"]["cache"]["files"] = [
                item["path"]
                for item in case_variant(case, variants)["files"]
                if not item["path"].endswith("/zarr.json")
            ]
        if "query_file" in public["result"]["scenario"]["sampling"]:
            public["result"]["scenario"]["sampling"]["query_file"] = (
                f"queries/{case['sampling']['query_id']}.bin"
            )
        observations.append(public)
    rows = []
    for case_id, values in samples.items():
        case = cases[case_id]
        variant = case_variant(case, variants)
        condition = (case["input_id"], case["backend"], case["workload"])
        if len(values) != plan["profile"]["repeats"]:
            raise ValueError("Missing repeated observations")
        reference = spread(references[condition])
        rows.append(
            {
                "id": case_id,
                "condition_id": identity([identity(context), *condition])[:20],
                "input_id": case["input_id"],
                "backend": case["backend"],
                "workload": case["workload"],
                "sampling": case["sampling"],
                **(
                    {"query_summary": case["query_summary"]}
                    if "query_summary" in case
                    else {}
                ),
                "setting": {
                    k: v for k, v in variant.items() if k not in {"files", "id"}
                },
                "compression_fold": variant["logical_bytes"] / variant["stored_bytes"],
                "metrics": {
                    key: spread([value["metrics"][key] for value in values])
                    for key in values[0]["metrics"]
                },
                "reference_useful_gib_s": reference,
                "reference_spread_percent": 100
                * (reference["max"] - reference["min"])
                / reference["median"],
                "observation_ids": [value["task"]["id"] for value in values],
            }
        )
    public_inputs = [
        {k: v for k, v in asset.items() if k != "path"} for asset in plan["inputs"]
    ]
    return {
        "version": 1,
        "id": identity(context)[:20],
        "label": plan["label"],
        "pilot": plan["pilot"],
        "context": context,
        "analysis": {
            "metrics_version": METRICS_VERSION,
            "script_sha256": file_hash(Path(__file__)),
        },
        "pipeline": plan["pipeline"],
        "profile": plan["profile"],
        "reference_setting": plan["reference_setting"],
        "preparation_packages": prepared["packages"],
        "inputs": public_inputs,
        **(
            {key: plan[key] for key in ("cohorts", "volume", "weighting")}
            if "cohorts" in plan
            else {}
        ),
        "rows": rows,
        "observations": observations,
    }


def publish(args):
    document = summarize(args.run)
    args.output.mkdir(parents=True, exist_ok=True)
    filename = f"{document['id']}.json"
    write_json(args.output / filename, document)
    from pareto_mixture import write_queries

    write_queries(read_json(args.run / "plan.json"), args.output)
    index_path = args.output / "index.json"
    index = (
        read_json(index_path) if index_path.exists() else {"version": 1, "studies": []}
    )
    entry = {
        "id": document["id"],
        "label": f"{document['label']} · {document['context']['machine']['host']} · {document['context']['started'][:10]}",
        "pilot": document["pilot"],
        "file": filename,
        "cache_policy": document["context"]["cache_policy"],
    }
    matches = [item for item in index["studies"] if item["id"] == entry["id"]]
    if matches and matches != [entry]:
        raise ValueError("Study index conflicts with the exported result")
    if not matches:
        index["studies"].append(entry)
        index["studies"].sort(
            key=lambda value: value.get("cache_policy") not in {"cold", "cold-pass"}
        )
        temporary = args.output / "index.json.tmp"
        with temporary.open("x") as stream:
            stream.write(json.dumps(index, indent=2) + "\n")
        temporary.replace(index_path)
    print(f"Published {len(document['rows'])} settings to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser(
        "plan", help="Read metadata and print or save the exact study matrix"
    )
    plan.add_argument("definition", type=Path)
    plan.add_argument("--corpus", type=Path, required=True)
    plan.add_argument("--pilot", action="store_true")
    plan.add_argument("--output", type=Path)
    prepare = commands.add_parser(
        "prepare", help="Verify inputs and encode every layout on a compute node"
    )
    prepare.add_argument("plan", type=Path)
    prepare.add_argument("--corpus", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    run = commands.add_parser(
        "run", help="Run repeated, randomized measurements on a compute node"
    )
    run.add_argument("plan", type=Path)
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--bench", type=Path, default=ROOT / "build/bench/damacy_bench")
    run.add_argument("--backend", choices=["cpu", "cuda"])
    run.add_argument(
        "--storage", required=True, help="Human-readable storage description"
    )
    report = commands.add_parser(
        "report", help="Validate and export a completed study for the report"
    )
    report.add_argument("run", type=Path)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            document = make_plan(
                read_json(args.definition),
                read_json(args.corpus / "manifest.json"),
                pilot=args.pilot,
            )
            if args.output:
                write_json(args.output, document)
            tasks = schedule(document)
            print(
                json.dumps(
                    {
                        "inputs": len(document["inputs"]),
                        "encoded_layouts": len(document["variants"]),
                        "logical_preparation_gib": sum(
                            math.prod(v["shape"]) * DTYPES[v["dtype"]][1]
                            for v in document["variants"]
                        )
                        / GIB,
                        "cohorts": len(document.get("cohorts", [])),
                        "settings": len(document["cases"]),
                        "executions": len(tasks),
                        "minimum_timed_seconds": sum(
                            document["profile"]["min_seconds"]
                            if t["role"] == "sample"
                            else document["profile"]["reference_seconds"]
                            for t in tasks
                        ),
                        "plan_sha256": identity(document),
                        "pilot": args.pilot,
                    },
                    indent=2,
                )
            )
        elif args.command == "prepare":
            require_compute()
            subprocess.run(
                [
                    "uv",
                    "run",
                    "--script",
                    str(ROOT / "bench/pareto_data.py"),
                    str(args.plan.resolve()),
                    "--corpus",
                    str(args.corpus.resolve()),
                    "--output",
                    str(args.output.resolve()),
                ],
                check=True,
            )
        elif args.command == "run":
            run_study(args)
        else:
            publish(args)
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
