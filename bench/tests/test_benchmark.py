"""Exercise completed passes, warmup accounting, and rejected read protocols."""

import copy
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def check(binary, backend):
    with tempfile.TemporaryDirectory(prefix="damacy-bench-test-") as temporary:
        root = Path(temporary)
        array = root / "array"
        array.mkdir()
        (array / "zarr.json").write_text(
            json.dumps(
                {
                    "zarr_format": 3,
                    "node_type": "array",
                    "shape": [2, 16, 16],
                    "data_type": "uint16",
                    "fill_value": 0,
                    "chunk_grid": {
                        "name": "regular",
                        "configuration": {"chunk_shape": [1, 16, 16]},
                    },
                    "chunk_key_encoding": {
                        "name": "default",
                        "configuration": {"separator": "/"},
                    },
                    "codecs": [
                        {"name": "bytes", "configuration": {"endian": "little"}}
                    ],
                    "attributes": {},
                }
            )
        )
        for plane in range(2):
            chunk = array / f"c/{plane}/0/0"
            chunk.parent.mkdir(parents=True)
            chunk.write_bytes(
                struct.pack("<256H", *range(plane * 256, (plane + 1) * 256))
            )
        scenario = {
            "dataset": {"store_root": str(root), "n_zarrs": 1, "uris": ["array"]},
            "sampling": {
                "pattern": "planes",
                "sample_shape": [1, 16, 16],
                "n_batches": 3,
                "n_warmup_batches": 1,
                "samples_per_batch": 2,
                "seed": 1234,
                "min_seconds": 0.03,
            },
            "pipeline": {
                "executor": backend,
                "dtype": "f32",
                "lookahead_samples": 8,
                "n_io_threads": 2,
                "decode_workers": 2,
                "chunks_per_input_buffer": 16,
                "max_cpu_memory_mb": 64,
                "max_gpu_memory_mb": 512,
                "max_chunk_uncompressed_mb": 1,
                "max_read_op_kb": 64,
                "max_substreams_per_chunk": 64,
            },
            "consumer": {"hold_ms": 0},
        }

        def run(value, valid=True):
            path = root / "scenario.json"
            path.write_text(json.dumps(value))
            result = subprocess.run(
                [binary, str(path)], text=True, capture_output=True, timeout=30
            )
            assert (result.returncode == 0) == valid, result.stderr
            return json.loads(result.stdout) if valid else result.stderr

        result = run(scenario)
        assert result["scenario"] == scenario
        assert result["measurement_passes"] > 0
        assert result["timings_ms"]["wall"] >= 30
        assert result["counters"]["batches_emitted"] == 3 * result["measurement_passes"]
        assert result["counters"]["reads_issued"] > 0
        assert result["derived"]["bytes_per_sample"] == 1024
        assert result["metadata"]["startup"]["read"] > 0
        assert result["metadata"]["measured"]["read"] == 0
        assert result["metadata"]["version"] == 2
        assert result["metadata"]["measured"]["array_metadata_misses"] == 0
        assert result["metadata"]["measured"]["shard_index_misses"] == 0
        assert "cold_reads" not in result

        one = copy.deepcopy(scenario)
        one["sampling"].update(
            pattern="random", sample_shape=[1, 8, 8], n_warmup_batches=0, min_seconds=0
        )
        result = run(one)
        assert result["measurement_passes"] == 1
        assert result["counters"]["batches_emitted"] == 3
        assert result["timings_ms"]["time_to_first_batch"] > 0
        assert result["metadata"]["measured"]["read"] > 0

        wrong = copy.deepcopy(scenario)
        wrong["sampling"]["sample_shape"] = [1, 8, 8]
        assert "planes requires" in run(wrong, False)
        wrong = copy.deepcopy(scenario)
        wrong["cache"] = {"policy": "cold", "files": [], "device_stats": []}
        assert "one batch per pass" in run(wrong, False)
        wrong["sampling"]["n_batches"] = 1
        assert "cold-read verification" in run(wrong, False)


if __name__ == "__main__":
    check(sys.argv[1], sys.argv[2])
