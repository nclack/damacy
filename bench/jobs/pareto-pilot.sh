#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?Run this script inside an approved compute allocation}"
study_mode=${1:?Expected cpu or gpu}
study_root=${2:?Expected a shared study directory containing source/ and plan.json}
study_corpus=${3:?Expected the version-3 microscopy corpus directory}
study_repo="$study_root/source"
study_scratch=$(mktemp -d /tmp/damacy-pareto.XXXXXXXX)
trap 'rm -rf "$study_scratch"' EXIT

export PATH="$HOME/.pixi/bin:$HOME/.pixi/envs/cuda-toolkit/bin:$PATH"
export CUDAToolkit_ROOT="$HOME/.pixi/envs/cuda-toolkit"
export Nvcomp_ROOT="$HOME/opt"
export CMAKE_PREFIX_PATH="$HOME/opt:$HOME/.pixi/envs/damacy-cpu-deps:$CUDAToolkit_ROOT"
export PKG_CONFIG_PATH="$HOME/opt/lib/pkgconfig:$HOME/.pixi/envs/damacy-cpu-deps/lib/pkgconfig"
export LD_LIBRARY_PATH="$HOME/opt/lib:$HOME/.pixi/envs/damacy-cpu-deps/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export UV_CACHE_DIR="$study_scratch/uv-cache"
export TMPDIR="$study_scratch"
cd "$study_repo"

case "$study_mode" in
  cpu)
    cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DDAMACY_CUDA=ON -DDAMACY_PYTHON=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=90-real
    cmake --build build --parallel 8
    uv run --no-project --python 3.12 python -m unittest discover -s bench -p test_pareto.py
    node --test bench/test_pareto.mjs
    uv run --no-project --with 'mkdocs>=1.6' --with 'mkdocs-material>=9.5' \
      --with 'mkdocstrings[python]>=0.26' --with 'ruff>=0.6' \
      mkdocs build --strict --site-dir "$study_root/site"
    uv run --script bench/pareto.py prepare "$study_root/plan.json" \
      --corpus "$study_corpus" --output "$study_root/data"
    ;;
  gpu)
    ctest --test-dir build --output-on-failure
    cp -a "$study_root/data" "$study_scratch/data"
    uv run --script bench/pareto.py run "$study_root/plan.json" \
      --data "$study_scratch/data" --output "$study_root/results" \
      --storage 'Local NVMe; file cache checked cold before every batch'
    cp -a docs/pareto "$study_root/report"
    uv run --script bench/pareto.py report "$study_root/results" \
      --output "$study_root/report/data"
    ;;
  *)
    printf 'Unknown job mode: %s\n' "$study_mode" >&2
    exit 2
    ;;
esac
