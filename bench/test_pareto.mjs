import assert from "node:assert/strict";
import test from "node:test";
import {csv, devicePlottable, frontier, ioPlottable, matches} from "../docs/pareto/analysis.mjs";

function point(id, fold, rate, condition = "cpu") {
  return {id, condition_id: condition, input_id: "image", backend: condition, workload: "crop",
    compression_fold: fold, setting: {codec: "zstd", chunk_bytes: 65536, block_bytes: null,
      chunk_shape: [1, 128, 256], shard_shape: [1, 1024, 1024]},
    metrics: {useful_gib_s: {median: rate, min: rate * .9, max: rate * 1.1, n: 3}}};
}

test("joint frontier keeps codec and size tradeoffs", () => {
  const rows = [point("fast", 1, 10), point("small", 5, 2), point("balanced", 3, 5), point("dominated", 2, 4)];
  assert.deepEqual(frontier(rows).map(row => row.id), ["fast", "balanced", "small"]);
});

test("different measurement conditions never dominate each other", () => {
  const rows = [point("cpu", 1, 1), point("gpu", 10, 10, "cuda")];
  assert.equal(frontier(rows).length, 2);
});

test("ties remain observable and invalid objectives are excluded", () => {
  const rows = [point("a", .5, 2), point("b", .5, 2), point("invalid", 0, 10), point("missing", 2, NaN)];
  assert.deepEqual(frontier(rows).map(row => row.id), ["a", "b"]);
});

test("Blosc block filters preserve raw controls", () => {
  const raw = point("raw", 2, 2), blosc = point("blosc", 3, 3);
  blosc.setting.codec = "blosc-zstd";
  blosc.setting.block_bytes = 65536;
  assert(matches(raw, {block: "16384"}));
  assert(!matches(blosc, {block: "16384"}));
  assert(matches(blosc, {block: "65536"}));
});

test("CSV keeps exact values and measurement identities", () => {
  const row = point("exact", 2.123456789012345, 4.123456789012345);
  row.input_id = 'quoted "image"';
  const output = csv([row]);
  assert(output.includes('"2.123456789012345"'));
  assert(output.includes('"4.123456789012345"'));
  assert(output.includes('"quoted ""image"""'));
  assert(output.includes('"cpu","exact"'));
});

test("application and device counters cannot substitute for each other", () => {
  const row = point("reads", 2, 4);
  assert(!ioPlottable(row));
  row.metrics.read_iops = {median: 12345.678901234};
  row.metrics.read_gib_s = {median: 1.2345678901234};
  assert(ioPlottable(row));
  assert(!devicePlottable(row));
  const output = csv([row]);
  assert(output.includes('"12345.678901234"'));
  assert(output.includes('"1.2345678901234"'));
  row.metrics.device_iops = {median: 17};
  row.metrics.device_read_gib_s = {median: 1};
  assert(devicePlottable(row));
  row.metrics.device_iops.median = NaN;
  assert(!devicePlottable(row));
});
