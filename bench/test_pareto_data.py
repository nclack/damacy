"""Verify tiled pixels and encoded shards with small, real Zarr arrays."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pareto_data import compressors, tiled_shard, write_volume

import zarr


class TilingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
