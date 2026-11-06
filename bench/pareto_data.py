#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.2", "zarr==3.3.0", "numcodecs==0.16.5"]
# ///
"""Encode and verify the exact microscopy layouts in a reader study."""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import random
import struct
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from pareto import (
    DTYPES,
    command_output,
    file_hash,
    identity,
    read_json,
    relative_path,
    require_compute,
    write_json,
)
from zarr.codecs import BloscCodec, ZstdCodec

import zarr


def input_path(root, asset):
    path = relative_path(root, asset["path"])
    expected = math.prod(asset["shape"]) * DTYPES[asset["dtype"]][1]
    if not path.is_file() or path.stat().st_size != expected:
        key = command_output(
            ["git", "-C", str(root), "annex", "lookupkey", "--", asset["path"]]
        )
        location = (
            command_output(["git", "-C", str(root), "annex", "contentlocation", key])
            if key
            else None
        )
        if location:
            parts = Path(location).parts
            if parts[0] == ".git":
                git_dir = command_output(
                    [
                        "git",
                        "-C",
                        str(root),
                        "rev-parse",
                        "--path-format=absolute",
                        "--git-dir",
                    ]
                )
                path = Path(git_dir).joinpath(*parts[1:]) if git_dir else path
            else:
                path = root / location
    if not path.is_file() or path.stat().st_size != expected:
        raise ValueError(
            f"Input is missing or has the wrong length: {asset['id']}; materialize the corpus first"
        )
    if file_hash(path) != asset["sha256"]:
        raise ValueError(f"Input checksum mismatch: {asset['id']}")
    return path


def compressors(variant):
    if variant["codec"] == "none":
        return []
    if variant["codec"] == "zstd":
        return [ZstdCodec(level=variant["level"], checksum=False)]
    return [
        BloscCodec(
            cname=variant["codec"].removeprefix("blosc-"),
            clevel=variant["level"],
            shuffle="bitshuffle",
            typesize=DTYPES[variant["dtype"]][1],
            blocksize=variant["block_bytes"],
        )
    ]


def chunk_records(array, variant):
    slots = math.prod(
        s // c
        for s, c in zip(variant["shard_shape"], variant["chunk_shape"], strict=True)
    )
    index_bytes = slots * 16 + 4
    blocks, payload_bytes, chunks = set(), 0, 0
    for file in sorted((array / "c").rglob("*")):
        if not file.is_file():
            continue
        size = file.stat().st_size
        with file.open("rb") as stream:
            stream.seek(-index_bytes, 2)
            index = stream.read(index_bytes)
            previous_end = 0
            for offset, length in struct.iter_unpack("<QQ", index[:-4]):
                if offset == 2**64 - 1 and length == 2**64 - 1:
                    if "baked" in variant:
                        raise ValueError("Baked samples must all be stored")
                    continue
                if offset + length > size - index_bytes:
                    raise ValueError("Invalid shard index")
                payload_bytes += length
                chunks += 1
                if "baked" in variant and offset != previous_end:
                    raise ValueError(
                        "Baked sample chunks must be contiguous in file order"
                    )
                previous_end = offset + length
                if variant["codec"].startswith("blosc-"):
                    stream.seek(offset)
                    header = stream.read(16)
                    decoded, block, stored = struct.unpack_from("<III", header, 4)
                    if (
                        decoded != variant["chunk_bytes"]
                        or not block
                        or stored != length
                    ):
                        raise ValueError(
                            "Blosc header disagrees with the requested chunk layout"
                        )
                    blocks.add(block)
    if not chunks:
        raise ValueError("No stored chunks were generated")
    return {
        "payload_bytes": payload_bytes,
        "stored_chunks": chunks,
        "padded_chunk_bytes": chunks * variant["chunk_bytes"],
        "actual_block_bytes": sorted(blocks),
        **({"contiguous_sample_chunks": True} if "baked" in variant else {}),
    }


def tiled_shard(pixels, variant, origin):
    return tiled_region(pixels, variant, origin, variant["shard_shape"])


def tiled_region(pixels, variant, origin, shape):
    if variant["tiling"]["method"] != "shifted-periodic-planes":
        raise ValueError("Unsupported tiling method")
    shape = tuple(shape)
    result = np.empty(shape, dtype=pixels.dtype)
    seed = int(identity([variant["input_id"], variant["tiling"]]), 16)
    for local_z in range(shape[0]):
        z = origin[0] + local_z
        rng = random.Random(seed + z)
        y = (
            origin[1] + np.arange(shape[1]) + rng.randrange(pixels.shape[1])
        ) % pixels.shape[1]
        x = (
            origin[2] + np.arange(shape[2]) + rng.randrange(pixels.shape[2])
        ) % pixels.shape[2]
        result[local_z] = pixels[z % pixels.shape[0]][np.ix_(y, x)]
    return result


def baked_shard(pixels, variant, start, coordinates):
    baked = variant["baked"]
    depth = variant["shard_shape"][0]
    shards = variant["shape"][0] // depth
    lane = start // depth
    result = np.empty(tuple(variant["shard_shape"]), dtype=pixels.dtype)
    source = {"input_id": variant["input_id"], "tiling": baked["tiling"]}
    for offset in range(depth):
        index = offset * shards + lane
        result[offset] = tiled_region(
            pixels, source, coordinates[index], baked["sample_shape"]
        )[0]
    return result


def write_volume(array, pixels, variant):
    if "baked" in variant:
        from pareto_crops import random_origins

        baked = variant["baked"]
        coordinates = random_origins(
            baked["source"],
            baked["seed"],
            baked["coordinate_key"],
            variant["shape"][0],
            baked["sample_shape"],
        )
        for start in range(0, variant["shape"][0], variant["shard_shape"][0]):
            selection = slice(start, start + variant["shard_shape"][0])
            expected = baked_shard(pixels, variant, start, coordinates)
            array[selection] = expected
            actual = array[selection]
            if not np.array_equal(actual.view(np.uint8), expected.view(np.uint8)):
                raise ValueError(
                    f"Baked shard does not round-trip exactly: {variant['id']} {start}"
                )
            del actual, expected
            print(f"  verified baked shard {start}", flush=True)
        return
    if "tiling" not in variant:
        array[:] = pixels
        if array[:].tobytes() != pixels.tobytes():
            raise ValueError("Encoded input does not round-trip exactly")
        return
    shape, shards = variant["shape"], variant["shard_shape"]
    for z in range(0, shape[0], shards[0]):
        for y in range(0, shape[1], shards[1]):
            for x in range(0, shape[2], shards[2]):
                origin = (z, y, x)
                selection = tuple(
                    slice(start, start + size)
                    for start, size in zip(origin, shards, strict=True)
                )
                expected = tiled_shard(pixels, variant, origin)
                array[selection] = expected
                actual = array[selection]
                if not np.array_equal(actual.view(np.uint8), expected.view(np.uint8)):
                    raise ValueError(
                        f"Encoded shard does not round-trip exactly: {variant['id']} {origin}"
                    )
                del actual, expected
                print(f"  verified shard {origin}", flush=True)


def prepare_variant(variant, asset, path, output, reuse=None):
    zarr.config.set({"async.concurrency": 8, "threading.max_workers": 8})
    destination = output / variant["id"]
    if (
        not destination.exists()
        and reuse
        and (reuse / variant["id"] / "layout.json").is_file()
    ):
        destination.symlink_to(
            (reuse / variant["id"]).resolve(), target_is_directory=True
        )
    if destination.exists():
        record = read_json(destination / "layout.json")
        if any(record[key] != value for key, value in variant.items()):
            raise ValueError("Existing layout differs from the study plan")
        for item in record["files"]:
            file = relative_path(output, item["path"])
            if (
                file.stat().st_size != item["bytes"]
                or file_hash(file) != item["sha256"]
            ):
                raise ValueError(f"Prepared file changed: {file}")
        return record
    pending = output / f"{variant['id']}.partial"
    pending.mkdir()
    dtype = np.dtype(asset["dtype"]).newbyteorder("<")
    pixels = np.memmap(path, dtype=dtype, mode="r", shape=tuple(asset["shape"]))
    array = zarr.create_array(
        store=str(pending / "array"),
        shape=tuple(variant["shape"]),
        dtype=dtype,
        chunks=tuple(variant["chunk_shape"]),
        shards=tuple(variant["shard_shape"]),
        compressors=compressors(variant),
        config={"write_empty_chunks": True},
        zarr_format=3,
    )
    write_volume(array, pixels, variant)
    details = chunk_records(pending / "array", variant)
    files = [
        {
            "path": f"{variant['id']}/{file.relative_to(pending).as_posix()}",
            "bytes": file.stat().st_size,
            "sha256": file_hash(file),
        }
        for file in sorted((pending / "array").rglob("*"))
        if file.is_file()
    ]
    record = {
        **variant,
        **details,
        "logical_bytes": math.prod(variant["shape"]) * dtype.itemsize,
        "shard_files": sum(not item["path"].endswith("/zarr.json") for item in files),
        "shard_file_bytes": [
            item["bytes"] for item in files if not item["path"].endswith("/zarr.json")
        ],
        "stored_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    write_json(pending / "layout.json", record)
    pending.rename(destination)
    return record


def prepare_variants(plan, inputs, paths, output, workers, reuse=None):
    if workers < 1:
        raise ValueError("Preparation needs at least one worker")
    arguments = [
        (
            variant,
            inputs[variant["input_id"]],
            paths[variant["input_id"]],
            output,
            reuse,
        )
        for variant in plan["variants"]
    ]
    if workers == 1:
        for args in arguments:
            yield prepare_variant(*args)
        return
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn")
    ) as executor:
        futures = [executor.submit(prepare_variant, *args) for args in arguments]
        try:
            for future in as_completed(futures):
                yield future.result()
        finally:
            for future in futures:
                future.cancel()


def prepare(plan, corpus, output, *, workers=1, reuse=None):
    require_compute()
    if identity(read_json(corpus / "manifest.json")) != plan["corpus_sha256"]:
        raise ValueError("Corpus metadata changed after planning")
    inputs = {asset["id"]: asset for asset in plan["inputs"]}
    paths = {name: input_path(corpus, asset) for name, asset in inputs.items()}
    packages = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "zarr", "numcodecs")
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "preparation.json",
        {"plan_sha256": identity(plan), "packages": packages},
    )
    from pareto_mixture import write_queries

    write_queries(plan, output)
    records = {}
    for number, record in enumerate(
        prepare_variants(plan, inputs, paths, output, workers, reuse), 1
    ):
        records[record["id"]] = record
        print(
            f"[{number}/{len(plan['variants'])}] {record['input_id']} {record['codec']} "
            f"{record['chunk_bytes']} / {record['block_bytes']}: "
            f"{record['logical_bytes'] / record['stored_bytes']:.3f}x",
            flush=True,
        )
    write_json(
        output / "prepared.json",
        {
            "version": 1,
            "plan_sha256": identity(plan),
            "packages": packages,
            "variants": [records[variant["id"]] for variant in plan["variants"]],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reuse", type=Path)
    args = parser.parse_args()
    prepare(
        read_json(args.plan),
        args.corpus,
        args.output,
        workers=args.workers,
        reuse=args.reuse,
    )


if __name__ == "__main__":
    main()
