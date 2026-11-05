#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?Run inside an approved compute allocation}"
study_backend=${1:?Expected cpu or cuda}
study_root=${2:?Expected shared study directory with source/ and plan.json}
study_corpus=${3:?Expected materialized microscopy corpus}
study_source="$study_root/source"
study_scratch=$(mktemp -d /tmp/damacy-mixture.XXXXXXXX)
trap 'rm -rf "$study_scratch"' EXIT

export PATH="$HOME/.pixi/bin:$HOME/.pixi/envs/cuda-toolkit/bin:$PATH"
export CUDAToolkit_ROOT="$HOME/.pixi/envs/cuda-toolkit"
export Nvcomp_ROOT="$HOME/opt"
export CMAKE_PREFIX_PATH="$HOME/opt:$HOME/.pixi/envs/damacy-cpu-deps:$CUDAToolkit_ROOT"
export PKG_CONFIG_PATH="$HOME/opt/lib/pkgconfig:$HOME/.pixi/envs/damacy-cpu-deps/lib/pkgconfig"
export LD_LIBRARY_PATH="$HOME/opt/lib:$HOME/.pixi/envs/damacy-cpu-deps/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export UV_CACHE_DIR="$study_scratch/uv-cache"
export TMPDIR="$study_scratch"
cd "$study_source"

uv run --no-project --python 3.12 python - "$study_scratch" <<'PY'
import json
import sys
from pathlib import Path
sys.path.insert(0, "bench")
from pareto import local_nvme_record
print(json.dumps(local_nvme_record(Path(sys.argv[1])), indent=2), flush=True)
PY

case "$study_backend" in
  cpu)
    cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo -DDAMACY_CUDA=ON -DDAMACY_PYTHON=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=90-real
    cmake --build build --parallel 16
    ctest --test-dir build --output-on-failure \
      -R '^(test_benchmark_cpu|test_cold_read|test_cpu_executor|test_cpu_pipeline|test_blosc_codecs|test_file_reader)$'
    uv run --script bench/pareto.py prepare "$study_root/plan.json" \
      --corpus "$study_corpus" --output "$study_root/data"
    cp -a "$study_root/data" "$study_scratch/data"
    study_backends=(--backend cpu)
    ;;
  cuda)
    study_backends=()
    ctest --test-dir build --output-on-failure
    cp -a "$study_root/data" "$study_scratch/data"
    ;;
  *)
    printf 'Expected cpu or cuda, got %s\n' "$study_backend" >&2
    exit 2
    ;;
esac

uv run --script bench/pareto.py run "$study_root/plan.json" \
  --data "$study_scratch/data" --output "$study_root/results-$study_backend" \
  "${study_backends[@]}" --storage 'Local NVMe; checked cold multi-batch passes'
uv run --script bench/pareto.py report "$study_root/results-$study_backend" \
  --output "$study_root/report/data"

