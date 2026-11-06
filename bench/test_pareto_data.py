"""Verify tiled pixels and encoded shards with small, real Zarr arrays."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pareto import file_hash, identity, read_json
from pareto_crops import baked_origins, random_origins
from pareto_data import chunk_records, compressors, prepare, tiled_shard, write_volume

import zarr


class TilingTests(unittest.TestCase):
    def test_baked_pixels_pair_with_cross_shard_crops_in_every_codec(self):
        bits = np.array(
            [0, 0x80000000, 0x7FC00001, 0x7F800000, 0x3F800000, 0xBF800000],
            dtype=np.uint32,
        )
        pixels = bits.view(np.float32).reshape(1, 2, 3)
        source = {
            "id": "example",
            "dtype": "float32",
            "shape": [4, 8, 8],
            "shard_shape": [4, 4, 4],
        }
        tiling = {"method": "shifted-periodic-planes", "seed": 42}
        original = tiled_shard(
            pixels,
            {"input_id": "example", "shard_shape": source["shape"], "tiling": tiling},
            (0, 0, 0),
        )
        sample = [1, 3, 3]
        coordinates = random_origins(source, 7, "crop-3", 20, sample)
        self.assertTrue(
            any(
                y // 4 != (y + 2) // 4 or x // 4 != (x + 2) // 4
                for _, y, x in coordinates
            )
        )
        for codec in ("none", "zstd", "blosc-zstd", "blosc-lz4"):
            with self.subTest(codec=codec), tempfile.TemporaryDirectory() as temporary:
                variant = {
                    "id": "baked",
                    "input_id": "example",
                    "dtype": "float32",
                    "codec": codec,
                    "shape": [20, 3, 3],
                    "shard_shape": [10, 3, 3],
                    "chunk_shape": sample,
                    "chunk_bytes": 36,
                    "block_bytes": 36 if codec.startswith("blosc") else None,
                    "level": 3,
                    "baked": {
                        "source": source,
                        "seed": 7,
                        "coordinate_key": "crop-3",
                        "sample_shape": sample,
                        "tiling": tiling,
                    },
                }
                path = Path(temporary) / "array"
                array = zarr.create_array(
                    store=str(path),
                    shape=variant["shape"],
                    chunks=sample,
                    shards=variant["shard_shape"],
                    dtype=pixels.dtype,
                    compressors=compressors(variant),
                    config={"write_empty_chunks": True},
                )
                write_volume(array, pixels, variant)
                for (z, y, x), (index, _, _) in zip(
                    coordinates, baked_origins(variant, 20), strict=True
                ):
                    self.assertEqual(
                        array[index].tobytes(),
                        original[z, y : y + 3, x : x + 3].tobytes(),
                    )
                self.assertTrue(
                    chunk_records(path, variant)["contiguous_sample_chunks"]
                )

    def test_spatial_seams_match_a_single_tiled_plane(self):
        pixels = np.arange(3 * 5 * 7, dtype=np.uint16).reshape(3, 5, 7)
        variant = {
            "input_id": "example",
            "shape": [4, 8, 8],
            "shard_shape": [4, 4, 4],
            "tiling": {"method": "shifted-periodic-planes", "seed": 42},
        }
        assembled = np.empty((4, 8, 8), dtype=pixels.dtype)
        for y in (0, 4):
            for x in (0, 4):
                assembled[:, y : y + 4, x : x + 4] = tiled_shard(
                    pixels, variant, (0, y, x)
                )
        whole = copy.deepcopy(variant)
        whole["shard_shape"] = [4, 8, 8]
        expected = tiled_shard(pixels, whole, (0, 0, 0))
        np.testing.assert_array_equal(assembled, expected)
        self.assertTrue(set(assembled.flat).issubset(set(pixels.flat)))

    def test_each_codec_round_trips_float_bits_and_same_pixels(self):
        values = np.array(
            [
                0,
                0x80000000,
                0x7FC00001,
                0x7F800000,
                0x3F800000,
                0xBF800000,
                0x40000000,
                0x40400000,
                0x40800000,
            ],
            dtype=np.uint32,
        )
        pixels = values.view(np.float32).reshape(1, 3, 3)
        expected = None
        for codec in ("none", "zstd", "blosc-zstd", "blosc-lz4"):
            with self.subTest(codec=codec), tempfile.TemporaryDirectory() as temporary:
                variant = {
                    "input_id": "example",
                    "dtype": "float32",
                    "codec": codec,
                    "shape": [4, 8, 8],
                    "shard_shape": [4, 4, 4],
                    "chunk_shape": [1, 2, 2],
                    "chunk_bytes": 16,
                    "block_bytes": 16,
                    "level": 3,
                    "tiling": {"method": "shifted-periodic-planes", "seed": 42},
                }
                array = zarr.create_array(
                    store=str(Path(temporary) / "array"),
                    shape=tuple(variant["shape"]),
                    dtype=pixels.dtype,
                    chunks=tuple(variant["chunk_shape"]),
                    shards=tuple(variant["shard_shape"]),
                    compressors=compressors(variant),
                    config={"write_empty_chunks": True},
                    zarr_format=3,
                )
                write_volume(array, pixels, variant)
                actual = array[:].view(np.uint32)
                if expected is None:
                    expected = actual
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(
                    len(
                        [
                            p
                            for p in (Path(temporary) / "array/c").rglob("*")
                            if p.is_file()
                        ]
                    ),
                    4,
                )

    def test_parallel_preparation_preserves_pixels_and_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pixels = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
            corpus = root / "corpus"
            corpus.mkdir()
            output = root / "data"
            source = corpus / "input.bin"
            source.write_bytes(pixels.tobytes())
            asset = {
                "id": "example",
                "dtype": "uint16",
                "shape": list(pixels.shape),
                "path": "input.bin",
                "sha256": file_hash(source),
            }
            manifest = {"inputs": [asset]}
            (corpus / "manifest.json").write_text(json.dumps(manifest))
            variants = [
                {
                    "id": codec,
                    "input_id": "example",
                    "dtype": "uint16",
                    "codec": codec,
                    "shape": [4, 8, 8],
                    "shard_shape": [4, 4, 4],
                    "chunk_shape": [1, 2, 2],
                    "chunk_bytes": 8,
                    "block_bytes": None,
                    "level": 3,
                    "tiling": {"method": "shifted-periodic-planes", "seed": 42},
                }
                for codec in ("none", "zstd")
            ]
            plan = {
                "variants": variants,
                "inputs": [asset],
                "corpus_sha256": identity(manifest),
            }
            with patch("pareto_data.require_compute"):
                prepare(plan, corpus, output, workers=2)
            prepared = read_json(output / "prepared.json")
            records = {record["id"]: record for record in prepared["variants"]}
            self.assertEqual(set(records), {variant["id"] for variant in variants})
            for variant in variants:
                record = records[variant["id"]]
                self.assertEqual(record["shard_files"], 4)
                actual = zarr.open_array(str(output / variant["id"] / "array"))[:]
                whole = {**variant, "shard_shape": variant["shape"]}
                np.testing.assert_array_equal(
                    actual, tiled_shard(pixels, whole, (0, 0, 0))
                )
                for file in record["files"]:
                    self.assertEqual(file_hash(output / file["path"]), file["sha256"])
            with patch("pareto_data.require_compute"):
                prepare(plan, corpus, output, workers=1)
            self.assertEqual(read_json(output / "prepared.json"), prepared)
            reused = root / "reused"
            with patch("pareto_data.require_compute"):
                prepare(plan, corpus, reused, workers=1, reuse=output)
            self.assertEqual(read_json(reused / "prepared.json"), prepared)
            self.assertTrue(all((reused / v["id"]).is_symlink() for v in variants))


if __name__ == "__main__":
    unittest.main()
