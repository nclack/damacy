# Benchmark measurements

The [read-performance report](../docs/performance.md) compares the retained H100
and EPYC crop studies. It uses `bench/pareto.py` and publishes to `docs/pareto/data`.
These studies do not run in the daily workflow.

The CI workflow (`.github/workflows/bench.yml`) runs daily at 07:23 UTC and on manual
`workflow_dispatch`. It builds the release image, runs the default and mixed
scenarios on the self-hosted GPU runner, and saves the results as a
`bench-measurements` artifact for 90 days. It does not publish history pages.
Old history URLs redirect to the read-performance report.

Metrics are prefixed by runner and scenario. The adapter retains separate files
for timings and throughput, so runs can be downloaded and compared by commit.
To measure a particular branch, dispatch the workflow against that ref.

## Adapter (for local use)

The same adapter that CI uses can be run by hand:

```sh
uv run bench/run.py bench/scenarios/default.json
latest=$(ls -1d bench/runs/default/*/ | tail -1)
uv run bench/to_gha_bench.py "${latest}results.json" \
    --runner "$(hostname)" \
    --out-smaller /tmp/smaller.json \
    --out-bigger  /tmp/bigger.json
```

Omit `--runner` if you only ever bench from one machine; including it
future-proofs the metric names.

Useful when iterating on which metrics to track — add or remove keys in
`bench/to_gha_bench.py` and re-run against an existing `results.json`
without re-running the bench itself.

## Adding scenarios

Add a `bench/scenarios/<name>.json` file and include its name in the workflow's
scenario loop. Dataset preparation is reused from the runner's persistent cache.
