"""Check source balancing, shard geometry, and paired sensitivity measurements."""

import copy
import math
import unittest
from collections import Counter
from pathlib import Path

from pareto import DTYPES, GIB, case_variant, make_plan, read_json, schedule
from pareto_mixture import query_records, source_queries
from test_pareto import fixture


def mixture_fixture():
    _, corpus = fixture()
    assets = corpus["datasets"][0]["assets"]
    corpus["datasets"] = [
        {"id": name, "version": 1, "modality": "test", "source": {}, "assets": values}
        for name, values in [
            ("opencell", assets[:2]),
            *[(asset["id"], [asset]) for asset in assets[2:]],
        ]
    ]
    definition = read_json(Path(__file__).parent / "studies/microscopy.json")
    return definition, corpus


class MixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.definition, cls.corpus = mixture_fixture()
        cls.plan = make_plan(cls.definition, cls.corpus, pilot=True)

    def test_pilot_keeps_every_source_and_all_omissions(self):
        self.assertEqual(len(self.plan["inputs"]), 7)
        self.assertEqual(len(self.plan["cohorts"]), 7)
        opencell = next(
            c for c in self.plan["cohorts"] if c["omitted_dataset"] == "opencell"
        )
        self.assertEqual(len(opencell["input_ids"]), 5)
        self.assertFalse(
            set(opencell["input_ids"]) & {"opencell-dna", "opencell-protein"}
        )

    def test_one_gib_shards_and_sixteen_spatial_files(self):
        for variant in self.plan["variants"]:
            bpe = DTYPES[variant["dtype"]][1]
            self.assertEqual(math.prod(variant["shard_shape"]) * bpe, GIB)
            self.assertEqual(math.prod(variant["shape"]) * bpe, 16 * GIB)
            self.assertEqual(variant["shape"][1:], [2048, 2048])
            self.assertEqual(variant["shard_shape"][1:], [512, 512])
            self.assertEqual(
                math.prod(variant["chunk_shape"]) * bpe, variant["chunk_bytes"]
            )
            self.assertTrue(
                all(
                    s % c == 0
                    for s, c in zip(
                        variant["shard_shape"], variant["chunk_shape"], strict=True
                    )
                )
            )

    def test_paired_queries_and_complete_batches(self):
        signatures = {}
        for case in self.plan["cases"]:
            signatures.setdefault((case["input_id"], case["workload"]), set()).add(
                case["sampling"]["query_sha256"]
            )
            self.assertGreater(case["sampling"]["n_batches"], 1)
            self.assertEqual(
                case["query_summary"]["samples"],
                case["sampling"]["n_batches"] * case["sampling"]["samples_per_batch"],
            )
        self.assertTrue(all(len(values) == 1 for values in signatures.values()))
        tasks = schedule(self.plan, "cpu")
        self.assertEqual(len(tasks), 308)

    def test_omission_only_removes_that_sources_queries(self):
        full = next(
            q for q in self.plan["query_sets"] if q["workload"] == "balanced-256"
        )
        omitted = next(
            q
            for q in self.plan["query_sets"]
            if q["workload"] == "balanced-256" and len(q["input_ids"]) == 5
        )
        names = omitted["input_ids"]
        expected = [
            (names.index(full["input_ids"][source]), *origin)
            for source, *origin in query_records(full)
            if full["input_ids"][source] in names
        ]
        self.assertEqual(list(query_records(omitted)), expected)

    def test_each_source_covers_all_shards_without_reusing_maximum_chunks(self):
        query = next(
            q for q in self.plan["query_sets"] if q["workload"] == "balanced-256"
        )
        for source in query["sources"]:
            coordinates = source_queries(source, query)
            chunks = [
                (z // source["plane_stride"], y // 512, x // 512)
                for z, y, x in coordinates
            ]
            self.assertEqual(len(set(chunks)), len(chunks))
            tiles = Counter((y // 512, x // 512) for _, y, x in coordinates)
            self.assertEqual(len(tiles), 16)
            self.assertEqual(set(tiles.values()), {query["samples_per_source"] // 16})
            self.assertTrue(
                all(y % 512 <= 256 and x % 512 <= 256 for _, y, x in coordinates)
            )
        self.assertEqual(query["summary"]["distinct_shards"], 112)
        self.assertEqual(query["summary"]["useful_bytes"], 15 * GIB // 4)

    def test_mixture_compression_sums_bytes(self):
        case = self.plan["cases"][0]
        variants = {
            v["id"]: {**v, "stored_bytes": 10 + i, "files": []}
            for i, v in enumerate(self.plan["variants"])
        }
        setting = case_variant(case, variants)
        self.assertEqual(setting["logical_bytes"], 112 * GIB)
        self.assertEqual(
            setting["stored_bytes"],
            sum(variants[key]["stored_bytes"] for key in case["variant_ids"]),
        )
        self.assertEqual(len(setting["sources"]), 7)

    def test_returned_byte_budget_requires_float32(self):
        definition = copy.deepcopy(self.definition)
        definition["pipeline"]["dtype"] = "u16"
        with self.assertRaisesRegex(ValueError, "float32 output"):
            make_plan(definition, self.corpus, pilot=True)

    def test_reject_nondividing_geometry_and_single_collection_omissions(self):
        definition = copy.deepcopy(self.definition)
        definition["volume"]["shard_spatial_shape"] = [511, 512]
        with self.assertRaisesRegex(ValueError, "divide"):
            make_plan(definition, self.corpus, pilot=True)
        _, corpus = fixture()
        with self.assertRaisesRegex(ValueError, "two dataset"):
            make_plan(self.definition, corpus, pilot=True)


if __name__ == "__main__":
    unittest.main()
