"""Paired random crops and sequential reads of the same baked pixels."""

from __future__ import annotations

import math
import random

from pareto import DTYPES, corpus_inputs, identity, positive_int, positive_number
from pareto_mixture import (
    bounded_chunk_shape,
    query_bytes,
    query_summary,
    source_geometry,
)


def random_origins(source, seed, key, count, sample):
    limits = [
        size - extent + 1 for size, extent in zip(source["shape"], sample, strict=True)
    ]
    if len(limits) != 3 or min(limits) <= 0:
        raise ValueError("Crop must fit inside the source volume")
    rng = random.Random(int(identity([seed, key, source["id"]]), 16))
    return [tuple(rng.randrange(limit) for limit in limits) for _ in range(count)]


def baked_origins(source, count):
    depth = source["shard_shape"][0]
    shards = source["shape"][0] // depth
    if count > source["shape"][0]:
        raise ValueError("Baked pass cannot exceed the stored sample count")
    return [(index % shards * depth + index // shards, 0, 0) for index in range(count)]


def shard_counts(inputs, total):
    positive_int(total, "active_shards")
    if total < len(inputs):
        raise ValueError("Baked reads need at least one shard per native-type source")
    weights = [DTYPES[asset["dtype"]][1] for asset in inputs]
    counts = [1] * len(inputs)
    for _ in range(total - len(inputs)):
        index = max(
            range(len(inputs)),
            key=lambda i: total * weights[i] - counts[i] * sum(weights),
        )
        counts[index] += 1
    return dict(zip((asset["id"] for asset in inputs), counts, strict=True))


def settings(definition, crop, access):
    result = []
    for requested in definition[f"{access}_settings"]:
        if crop not in requested.get("crop_sizes", [crop]):
            continue
        codec = requested["codec"]
        if codec not in {"none", "zstd", "blosc-zstd", "blosc-lz4"}:
            raise ValueError(f"Unsupported reader codec: {codec}")
        size = (
            positive_int(requested["chunk_bytes"], "chunk_bytes")
            if access == "random"
            else None
        )
        block = requested["block_bytes"]
        if codec.startswith("blosc-"):
            if block != "chunk":
                positive_int(block, "block_bytes")
                if size and block > size:
                    raise ValueError("Blosc block cannot exceed the chunk")
        elif block is not None:
            raise ValueError("Only Blosc accepts a block size")
        setting = {
            "codec": codec,
            "chunk_mode": "bytes" if size else "sample",
            "chunk_bytes": size,
            "block_mode": "chunk" if block == "chunk" else "bytes" if block else "none",
            "block_bytes": None if block == "chunk" else block,
            "level": 0 if codec == "none" else definition["level"],
            "shuffle": "bitshuffle" if codec.startswith("blosc-") else "none",
        }
        result.append({**setting, "recipe_id": identity(setting)[:20]})
    if not result or len({s["recipe_id"] for s in result}) != len(result):
        raise ValueError("Layouts must be nonempty and distinct")
    return result


def variant_for(asset, source, setting, definition, baked=None):
    itemsize = DTYPES[asset["dtype"]][1]
    chunks = (
        [1, *source["shape"][1:]]
        if baked
        else bounded_chunk_shape(
            setting["chunk_bytes"], itemsize, source["shard_shape"]
        )
    )
    size = math.prod(chunks) * itemsize
    block = (
        size
        if setting["block_mode"] == "chunk"
        else min(size, setting["block_bytes"])
        if setting["block_bytes"]
        else None
    )
    if size > definition["pipeline"]["max_chunk_uncompressed_mb"] * 1024**2:
        raise ValueError("Chunk exceeds the reader's uncompressed byte limit")
    if (
        block
        and math.ceil(size / block) > definition["pipeline"]["max_substreams_per_chunk"]
    ):
        raise ValueError("Blosc blocks exceed max_substreams_per_chunk")
    variant = {
        "codec": setting["codec"],
        "chunk_bytes": size,
        "block_bytes": block,
        "input_id": asset["id"],
        "input_sha256": asset["sha256"],
        "dtype": asset["dtype"],
        "source_shape": asset["shape"],
        "shape": source["shape"],
        "chunk_shape": chunks,
        "shard_shape": source["shard_shape"],
        "level": setting["level"],
        "shuffle": setting["shuffle"],
        **(
            {"baked": baked}
            if baked
            else {
                "tiling": {
                    "method": "shifted-periodic-planes",
                    "seed": definition["volume"]["seed"],
                }
            }
        ),
    }
    return {"id": identity(variant)[:20], **variant}


def make_crop_plan(definition, corpus, *, pilot=False):
    if corpus["version"] != definition["corpus_version"]:
        raise ValueError("Corpus version does not match the study definition")
    if definition["cache_policy"] != "cold-pass":
        raise ValueError("Crop comparisons require checked cold passes")
    if definition["pipeline"]["dtype"] != "f32":
        raise ValueError(
            "Crop comparisons require float32 output for their byte budgets"
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
    crops = definition["crop_sizes"]
    if not crops or len(crops) != len(set(crops)):
        raise ValueError("Crop sizes must be nonempty and distinct")
    for crop in crops:
        positive_int(crop, "crop size")
    batch_bytes = positive_int(definition["batch_output_bytes"], "batch_output_bytes")
    lookahead_bytes = positive_int(
        definition["lookahead_output_bytes"], "lookahead_output_bytes"
    )
    if lookahead_bytes < batch_bytes:
        raise ValueError("Lookahead must hold at least one batch")
    sources = []
    for asset in inputs:
        shape, shards = source_geometry(asset, definition["volume"])
        sources.append(
            {
                "id": asset["id"],
                "dtype": asset["dtype"],
                "shape": shape,
                "shard_shape": shards,
            }
        )
    lanes = shard_counts(inputs, definition["baked"]["active_shards"])
    shard_bytes = positive_int(definition["baked"]["shard_bytes"], "baked shard_bytes")
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
    variants, queries, cases = {}, [], []
    for crop in crops:
        sample = [1, crop, crop]
        sample_bytes = math.prod(sample) * 4
        batch, remainder = divmod(batch_bytes, sample_bytes)
        lookahead, lookahead_remainder = divmod(lookahead_bytes, sample_bytes)
        if not batch or remainder or lookahead_remainder:
            raise ValueError(
                "Output byte budgets must contain whole samples of every crop size"
            )
        cache_samples = lookahead + 2 * batch
        for key, required in (
            ("n_array_meta_cache", cache_samples),
            ("n_chunk_layout_cache", cache_samples),
            (
                "n_shard_index_cache",
                cache_samples
                * positive_int(
                    definition["pipeline"]["max_shards_per_sample"],
                    "max_shards_per_sample",
                ),
            ),
        ):
            if positive_int(definition["pipeline"][key], key) < required:
                raise ValueError(f"{key} must be at least {required} for crop {crop}")
        count = (
            math.ceil(profile["returned_bytes_per_source"] / (sample_bytes * batch))
            * batch
        )
        multiple = math.lcm(batch, *lanes.values())
        corpus_count = (
            math.ceil(
                shard_bytes
                * sum(lanes.values())
                / (
                    crop
                    * crop
                    * sum(DTYPES[asset["dtype"]][1] for asset in inputs)
                    * multiple
                )
            )
            * multiple
        )
        if corpus_count < count:
            raise ValueError("Baked corpus must hold a complete measurement pass")
        baked_sources = [
            {
                "id": source["id"],
                "dtype": source["dtype"],
                "shape": [corpus_count, crop, crop],
                "shard_shape": [corpus_count // lanes[source["id"]], crop, crop],
            }
            for source in sources
        ]
        coordinate_key = f"crop-{crop}"
        for access, geometry in (("random", sources), ("baked", baked_sources)):
            recipes = settings(definition, crop, access)
            by_recipe = {}
            for setting in recipes:
                by_recipe[setting["recipe_id"]] = {}
                for asset, source, original in zip(
                    inputs, geometry, sources, strict=True
                ):
                    baked = (
                        {
                            "seed": definition["query_seed"],
                            "coordinate_key": coordinate_key,
                            "sample_shape": sample,
                            "source": original,
                            "tiling": {
                                "method": "shifted-periodic-planes",
                                "seed": definition["volume"]["seed"],
                            },
                        }
                        if access == "baked"
                        else None
                    )
                    variant = variant_for(asset, source, setting, definition, baked)
                    variants[variant["id"]] = variant
                    by_recipe[setting["recipe_id"]][asset["id"]] = variant["id"]
            for cohort in cohorts:
                query = {
                    "seed": definition["query_seed"],
                    "workload": f"{access}-{crop}",
                    "method": "random-translations"
                    if access == "random"
                    else "baked-sequential",
                    "coordinate_key": coordinate_key,
                    "input_ids": cohort["input_ids"],
                    "sources": geometry,
                    "samples_per_source": count,
                    "sample_shape": sample,
                    "samples_per_batch": batch,
                    "lookahead_samples": lookahead,
                }
                query = {"id": identity(query)[:20], **query}
                data = query_bytes(query)
                query["summary"] = query_summary(query, data)
                if access == "baked":
                    paired = next(
                        q
                        for q in queries
                        if q["workload"] == f"random-{crop}"
                        and q["input_ids"] == cohort["input_ids"]
                    )
                    query["source_query_id"] = paired["id"]
                    query["source_query_sha256"] = paired["summary"]["sha256"]
                    query["summary"]["source_query_id"] = paired["id"]
                    query["summary"]["source_query_sha256"] = paired["summary"][
                        "sha256"
                    ]
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
                for setting in recipes:
                    ids = [
                        by_recipe[setting["recipe_id"]][name]
                        for name in cohort["input_ids"]
                    ]
                    for backend in definition["backends"]:
                        case = {
                            "input_id": cohort["id"],
                            "variant_id": identity(ids)[:20],
                            "variant_ids": ids,
                            "backend": backend,
                            "workload": query["workload"],
                            "access_pattern": access,
                            "crop_size": crop,
                            "setting": setting,
                            "sampling": sampling,
                            "pipeline": {"lookahead_samples": lookahead},
                            "query_summary": query["summary"],
                        }
                        cases.append({"id": identity(case)[:20], **case})
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Measurement cases must be distinct")
    return {
        "version": 3,
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
        "baked": {**definition["baked"], "source_shards": lanes},
        "batch_output_bytes": batch_bytes,
        "lookahead_output_bytes": lookahead_bytes,
        "variants": list(variants.values()),
        "cases": cases,
        "query_sets": queries,
        "profile": profile,
        "pipeline": definition["pipeline"],
        "order_seed": definition["order_seed"],
        "reference_setting": definition["reference_setting"],
        "cache_policy": "cold-pass",
        "weighting": "Equal sample counts per input; original source byte sizes",
    }
