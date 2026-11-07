"""Tiled microscopy sources and paired, shard-balanced query streams."""

from __future__ import annotations

import hashlib
import math
import random
import statistics
import struct
from collections import Counter
from pathlib import Path

from pareto import DTYPES, corpus_inputs, identity, positive_int, positive_number


def source_geometry(asset, volume):
    frame = volume["frame_shape"]
    spatial = volume["shard_spatial_shape"]
    if len(frame) != 2 or len(spatial) != 2:
        raise ValueError("Frame and spatial shard shapes must have two axes")
    for size, shard in zip(frame, spatial, strict=True):
        positive_int(size, "frame dimension")
        positive_int(shard, "spatial shard dimension")
        if size % shard:
            raise ValueError("Spatial shards must divide the frame exactly")
    shard_bytes = positive_int(volume["shard_bytes"], "shard_bytes")
    depth, remainder = divmod(
        shard_bytes, math.prod(spatial) * DTYPES[asset["dtype"]][1]
    )
    if remainder or not depth:
        raise ValueError("Shard bytes must hold a whole number of spatial planes")
    return [depth, *frame], [depth, *spatial]


def bounded_chunk_shape(byte_count, itemsize, shards):
    elements, remainder = divmod(positive_int(byte_count, "chunk_bytes"), itemsize)
    if remainder or not elements or elements & (elements - 1):
        raise ValueError("Chunks must contain a power-of-two number of pixels")
    shape = [1, 1, 1]
    while math.prod(shape) < elements:
        axes = sorted((1, 2), key=lambda axis: (shape[axis], -axis))
        axis = next((a for a in axes if 2 * shape[a] <= shards[a]), 0)
        shape[axis] *= 2
        if shape[axis] > shards[axis] or shards[axis] % shape[axis]:
            raise ValueError("Chunk shape must divide the shard shape")
    return shape


def layouts(definition, pilot):
    result = []
    for size in definition["chunk_bytes"]:
        positive_int(size, "chunk_bytes")
        for codec in definition["codecs"]:
            if codec not in {"none", "zstd", "blosc-zstd", "blosc-lz4"}:
                raise ValueError(f"Unsupported reader codec: {codec}")
            blocks = [None]
            if codec.startswith("blosc-"):
                blocks = sorted(
                    {
                        size if b == "chunk" else positive_int(b, "block_bytes")
                        for b in definition["blosc_block_bytes"]
                    }
                )
                blocks = [b for b in blocks if b <= size]
            for block in blocks:
                setting = {"codec": codec, "chunk_bytes": size, "block_bytes": block}
                if pilot and setting not in definition["pilot_settings"]:
                    continue
                if (
                    block
                    and math.ceil(size / block)
                    > definition["pipeline"]["max_substreams_per_chunk"]
                ):
                    raise ValueError("Blosc blocks exceed max_substreams_per_chunk")
                result.append(setting)
    if not result or len({identity(s) for s in result}) != len(result):
        raise ValueError("Layouts must be nonempty and distinct")
    return result


def source_queries(source, query):
    if query.get("method") == "random-translations":
        from pareto_crops import random_origins

        return random_origins(
            source,
            query["seed"],
            query["coordinate_key"],
            query["samples_per_source"],
            query["sample_shape"],
        )
    if query.get("method") == "baked-sequential":
        from pareto_crops import baked_origins

        return baked_origins(source, query["samples_per_source"])
    rng = random.Random(
        int(identity([query["seed"], query["workload"], source["id"]]), 16)
    )
    depth, height, width = source["shape"]
    _, shard_y, shard_x = source["shard_shape"]
    stride = source["plane_stride"]
    planes = list(range(0, depth, stride))
    count = query["samples_per_source"]
    if query["workload"] == "whole-planes":
        if count > len(planes):
            raise ValueError("Whole-plane pass would revisit a maximum-size chunk")
        rng.shuffle(planes)
        return [(z + rng.randrange(stride), 0, 0) for z in planes[:count]]
    _, sample_y, sample_x = query["sample_shape"]
    if sample_y > shard_y or sample_x > shard_x:
        raise ValueError("Balanced crops must fit inside a spatial shard")
    tiles = [
        (y, x) for y in range(0, height, shard_y) for x in range(0, width, shard_x)
    ]
    if count > len(planes) * len(tiles):
        raise ValueError("Crop pass would revisit a maximum-size chunk")
    orders = {}
    for tile in tiles:
        order = planes.copy()
        rng.shuffle(order)
        orders[tile] = order
    queries = []
    while len(queries) < count:
        rng.shuffle(tiles)
        for y, x in tiles:
            if len(queries) == count:
                break
            z = orders[(y, x)].pop() + rng.randrange(stride)
            queries.append(
                (
                    z,
                    y + rng.randrange(shard_y - sample_y + 1),
                    x + rng.randrange(shard_x - sample_x + 1),
                )
            )
    return queries


def query_records(query):
    sources = query["sources"]
    active = {name: i for i, name in enumerate(query["input_ids"])}
    coordinates = {source["id"]: source_queries(source, query) for source in sources}
    rng = random.Random(query["seed"])
    names = [source["id"] for source in sources]
    for index in range(query["samples_per_source"]):
        rng.shuffle(names)
        for name in names:
            if name in active:
                yield (active[name], *coordinates[name][index])


def query_bytes(query):
    count = query["samples_per_source"] * len(query["input_ids"])
    return (
        b"DAMACYQ1"
        + struct.pack("<Q", count)
        + b"".join(struct.pack("<4I", *record) for record in query_records(query))
    )


def query_summary(query, data):
    sources = {s["id"]: s for s in query["sources"]}
    selected = [sources[name] for name in query["input_ids"]]
    sample_pixels = math.prod(query["sample_shape"])
    counts = Counter()
    windows = {
        "batch": (query["samples_per_batch"], set(), []),
        "lookahead": (query["lookahead_samples"], set(), []),
    }
    all_shards = set()
    crossing = 0
    records = list(struct.iter_unpack("<4I", data[16:]))
    for number, (source_id, *origin) in enumerate(records, 1):
        source = selected[source_id]
        counts[source["id"]] += 1
        shards = source["shard_shape"]
        last = [
            start + size - 1
            for start, size in zip(origin, query["sample_shape"], strict=True)
        ]
        touched = {
            (source_id, z, y, x)
            for z in range(origin[0] // shards[0], last[0] // shards[0] + 1)
            for y in range(origin[1] // shards[1], last[1] // shards[1] + 1)
            for x in range(origin[2] // shards[2], last[2] // shards[2] + 1)
        }
        all_shards.update(touched)
        crossing += len(touched) > 1
        for size, pending, samples in windows.values():
            pending.update(touched)
            if number % size == 0 or number == len(records):
                samples.append(len(pending))
                pending.clear()
    composition = [
        {
            "input_id": source["id"],
            "samples": counts[source["id"]],
            "useful_bytes": counts[source["id"]]
            * sample_pixels
            * DTYPES[source["dtype"]][1],
        }
        for source in selected
    ]
    return {
        **({"samples_crossing_shards": crossing} if "method" in query else {}),
        "sha256": hashlib.sha256(data).hexdigest(),
        "samples": len(records),
        "useful_bytes": sum(item["useful_bytes"] for item in composition),
        "source_samples": composition,
        "distinct_shards": len(all_shards),
        "shards_per_window": {
            name: {
                "min": min(values),
                "median": statistics.median(values),
                "max": max(values),
            }
            for name, (_, _, values) in windows.items()
        },
    }


def make_mixture_plan(definition, corpus, *, pilot=False):
    if corpus["version"] != definition["corpus_version"]:
        raise ValueError("Corpus version does not match the study definition")
    if definition["cache_policy"] != "cold-pass":
        raise ValueError("Mixed studies require checked cold passes")
    if definition["pipeline"]["dtype"] != "f32":
        raise ValueError(
            "Mixed studies require float32 output for their returned-byte budget"
        )
    names = definition["inputs"]
    if names == "all":
        names = [
            asset["id"] for dataset in corpus["datasets"] for asset in dataset["assets"]
        ]
    if not names or len(names) != len(set(names)):
        raise ValueError("Input IDs must be nonempty and distinct")
    inputs = corpus_inputs(corpus, names)
    profile = definition["pilot_profile"] if pilot else definition["profile"]
    for key in ("repeats", "returned_bytes_per_source", "warmup_batches"):
        positive_int(profile[key], key)
    for key in ("min_seconds", "reference_seconds", "timeout_seconds"):
        positive_number(profile[key], key)
    if max(profile["min_seconds"], profile["reference_seconds"]) > 600:
        raise ValueError("Measurement windows must be at most 600 seconds")
    if profile["timeout_seconds"] <= max(
        profile["min_seconds"], profile["reference_seconds"]
    ):
        raise ValueError("Process timeout must exceed the measurement window")
    positive_int(definition["query_seed"], "query_seed")
    if not definition["backends"] or set(definition["backends"]) - {"cpu", "cuda"}:
        raise ValueError("Backends must be cpu or cuda")
    if len(set(definition["backends"])) != len(definition["backends"]):
        raise ValueError("Duplicate backend")
    settings = layouts(definition, pilot)
    variants, by_setting, sources = [], {}, []
    for asset in inputs:
        shape, shards = source_geometry(asset, definition["volume"])
        itemsize = DTYPES[asset["dtype"]][1]
        sources.append(
            {
                "id": asset["id"],
                "dtype": asset["dtype"],
                "shape": shape,
                "shard_shape": shards,
                "plane_stride": max(
                    bounded_chunk_shape(size, itemsize, shards)[0]
                    for size in definition["chunk_bytes"]
                ),
            }
        )
        for setting in settings:
            variant = {
                **setting,
                "input_id": asset["id"],
                "input_sha256": asset["sha256"],
                "dtype": asset["dtype"],
                "source_shape": asset["shape"],
                "shape": shape,
                "chunk_shape": bounded_chunk_shape(
                    setting["chunk_bytes"], itemsize, shards
                ),
                "shard_shape": shards,
                "level": 0 if setting["codec"] == "none" else definition["level"],
                "shuffle": "bitshuffle"
                if setting["codec"].startswith("blosc-")
                else "none",
                "tiling": {
                    "method": "shifted-periodic-planes",
                    "seed": definition["volume"]["seed"],
                },
            }
            variant = {"id": identity(variant)[:20], **variant}
            variants.append(variant)
            by_setting.setdefault(identity(setting), {})[asset["id"]] = variant["id"]
    cohorts = [
        {
            "id": "all-sources",
            "name": "All sources",
            "input_ids": names,
            "omitted_dataset": None,
        }
    ]
    if definition["leave_one_out"]:
        for dataset in dict.fromkeys(asset["dataset_id"] for asset in inputs):
            included = [
                asset["id"] for asset in inputs if asset["dataset_id"] != dataset
            ]
            if not included:
                raise ValueError(
                    "Leave-one-out requires at least two dataset collections"
                )
            cohorts.append(
                {
                    "id": f"without-{dataset}",
                    "name": f"Without {dataset}",
                    "input_ids": included,
                    "omitted_dataset": dataset,
                }
            )
    cases, queries = [], []
    for cohort in cohorts:
        for pattern in definition["patterns"]:
            if pattern["id"] not in {"balanced-256", "whole-planes"}:
                raise ValueError("Mixed workloads must be balanced-256 or whole-planes")
            sample = (
                [1, *definition["volume"]["frame_shape"]]
                if pattern["id"] == "whole-planes"
                else pattern["sample_shape"]
            )
            if (
                len(sample) != 3
                or any(type(n) is not int or n <= 0 for n in sample)
                or sample[0] != 1
            ):
                raise ValueError(
                    "Mixed samples must have one plane and positive spatial dimensions"
                )
            batch = positive_int(pattern["samples_per_batch"], "samples_per_batch")
            count = (
                math.ceil(
                    profile["returned_bytes_per_source"]
                    / (math.prod(sample) * 4 * batch)
                )
                * batch
            )
            query = {
                "seed": definition["query_seed"],
                "workload": pattern["id"],
                "input_ids": cohort["input_ids"],
                "sources": sources,
                "samples_per_source": count,
                "sample_shape": sample,
                "samples_per_batch": batch,
                "lookahead_samples": definition["pipeline"]["lookahead_samples"],
            }
            query = {"id": identity(query)[:20], **query}
            data = query_bytes(query)
            query["summary"] = query_summary(query, data)
            queries.append(query)
            sampling = {
                "pattern": "queries",
                "query_id": query["id"],
                "query_sha256": query["summary"]["sha256"],
                "sample_shape": sample,
                "samples_per_batch": batch,
                "n_batches": count * len(cohort["input_ids"]) // batch,
                "n_warmup_batches": profile["warmup_batches"],
                "min_seconds": profile["min_seconds"],
                "seed": query["seed"],
            }
            if sampling["n_warmup_batches"] > sampling["n_batches"]:
                raise ValueError("Warmup must fit in the retained query stream")
            if definition["pipeline"]["lookahead_samples"] < batch:
                raise ValueError("Lookahead must hold at least one batch")
            for setting in settings:
                ids = [
                    by_setting[identity(setting)][name] for name in cohort["input_ids"]
                ]
                for backend in definition["backends"]:
                    case = {
                        "input_id": cohort["id"],
                        "variant_id": identity(ids)[:20],
                        "variant_ids": ids,
                        "backend": backend,
                        "workload": pattern["id"],
                        "sampling": sampling,
                        "query_summary": query["summary"],
                    }
                    cases.append({"id": identity(case)[:20], **case})
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Measurement cases must be distinct")
    return {
        "version": 2,
        "id": definition["id"],
        "label": definition["label"],
        "pilot": pilot,
        "definition_sha256": identity(definition),
        "corpus_id": corpus["id"],
        "corpus_version": corpus["version"],
        "corpus_sha256": identity(corpus),
        "inputs": inputs,
        "cohorts": cohorts,
        "volume": definition["volume"],
        "variants": variants,
        "cases": cases,
        "query_sets": queries,
        "profile": profile,
        "pipeline": definition["pipeline"],
        "order_seed": definition["order_seed"],
        "reference_setting": definition["reference_setting"],
        "cache_policy": "cold-pass",
        "weighting": "Equal sample counts per input; original source byte sizes",
    }


def write_queries(plan, output):
    for query in plan.get("query_sets", []):
        data = query_bytes(query)
        if hashlib.sha256(data).hexdigest() != query["summary"]["sha256"]:
            raise ValueError("Regenerated query stream differs from the plan")
        path = Path(output) / "queries" / f"{query['id']}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError("Retained queries changed")
        else:
            with path.open("xb") as stream:
                stream.write(data)
