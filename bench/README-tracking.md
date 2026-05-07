# Benchmark tracking

Wall-clock and throughput numbers from `damacy_bench` are tracked over
time on the `gh-pages` branch via
[github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark).

The CI workflow (`.github/workflows/bench.yml`) runs daily at 07:23 UTC
and on manual `workflow_dispatch`. It does **not** run on every push or
PR — full bench takes ~30 min and per-merge data points are noisier
than they are useful. To compare a specific PR or commit, dispatch the
workflow manually from the Actions tab against that ref.

1. Builds the release docker image.
2. Runs `bench/scenarios/default.json` then `bench/scenarios/mixed.json`
   inside the container, against the self-hosted GPU runner. The
   `bench/data/` tree is bind-mounted from the runner host so dataset
   generation is amortized across runs.
3. Pipes each `results.json` through `bench/to_gha_bench.py`, which
   flattens the rich bench output into the
   `{name, unit, value}` array shape the action expects. Metrics are
   prefixed with `<runner-name>/<scenario>/` (e.g.
   `gpu-host-1/default/throughput`,
   `gpu-host-1/mixed/decompress.ms_avg`) so histories from different
   machines stay separable on the dashboard.
4. Publishes two files — one per direction (smaller-is-better timings,
   bigger-is-better throughput) — to the `gh-pages` branch.

## Dashboard

Once the workflow has run once, the chart lives at

    https://<owner>.github.io/<repo>/bench/timings/
    https://<owner>.github.io/<repo>/bench/throughput/

Each metric becomes its own chart; PRs that touch `main` get a
regression comment if any metric exceeds `alert-threshold` (currently
150% — loose, intended to be tightened once the noise floor is known).

## One-time setup

1. **Create the `gh-pages` branch** (the action populates it but the
   branch must exist):
   ```sh
   git checkout --orphan gh-pages
   git rm -rf .
   echo "# damacy benchmarks" > README.md
   git add README.md
   git commit -m "init gh-pages"
   git push origin gh-pages
   git checkout main
   ```
2. **Settings → Pages**: source = `gh-pages` branch, `/` root.
3. **Settings → Actions → General → Workflow permissions**: confirm
   "Read and write" is enabled. (The workflow also asserts this via its
   `permissions:` block.)

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

## Tuning

- **`alert-threshold`** — loose now (150%), intended to drop to ~110%
  once 20+ data points are in.
- **`max-items-in-chart`** — capped at 200 to keep chart pages snappy;
  the underlying `data.js` retains full history.
- **Adding scenarios** — drop another `bench/scenarios/<name>.json`,
  add it to the `for scenario in …` loop in `.github/workflows/bench.yml`.
- **Multiple runners** — already handled: each metric is prefixed with
  `${{ runner.name }}`, so a second self-hosted runner produces a
  parallel set of charts (`runner-a/default/throughput`,
  `runner-b/default/throughput`) without trampling existing history.
