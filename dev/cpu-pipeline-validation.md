# CPU pipeline validation — 2026-09-14

The CPU executor uses ordinary host memory and shares rectangular query
preparation with the CUDA executor. The construction and lifetime contract is
in [Pipeline composition](../docs/pipeline.md); the extension plan for index
queries and NGFF resampling is in [CPU pipeline and query architecture](cpu-pipeline.md).

## Workload

The existing throughput dataset contains 16 sharded Zarr v3 arrays of shape
`512 × 512 × 512`, with `u16` data and zstd compression. Chunks are
`32 × 64 × 64`; shards are `256 × 256 × 512`. Data generation uses seed 42;
crop selection uses seed 1234. Each batch contains 128 crops of shape
`128 × 128 × 128`, converted to `f32`: 1 GiB of useful output per batch.

The CPU measurements use three warmup batches and twelve measured batches,
eight I/O workers, and a 4 GiB executor memory limit. Each worker count was
measured once in the same 16-core CPU allocation, in the order 1, 4, 8, 16.
Filesystem caches were shared between runs. These are rates after warmup, not
cold-storage measurements or training-loop timings. The scenario
is [throughput-cpu.json](../bench/scenarios/throughput-cpu.json). It now sets
a 5 GiB limit, because later 256-chunk input buffers reserve 2 GiB with the
default 4 MiB encoded-chunk bound.

CUDA comparisons use five warmup batches and thirty measured batches, with a
6 GiB device-memory limit. Baseline and refactor runs alternate on one L40
with eight allocated CPU cores. Both harnesses initialize the CUDA context
before creating the pipeline and use the legacy captured-context API. Both
builds use `RelWithDebInfo` with the same C/C++ and CUDA compilers. The baseline
is an independent source snapshot of commit
`dee3e660c58009a1ae22a3774c963e16c9e086db`, compiled for `sm_89` with
the same CUDA 13.1.2 toolkit and nvCOMP 5.2.0.10. The CPU build uses GCC 11.4,
zstd 1.5.7, and C-Blosc 1.21.6. Builds and CPU tests run on CPU compute nodes.

## CPU results

Throughput is useful output bytes divided by measured wall time; GB/s is
decimal. Peak RAM is the benchmark process's maximum resident set size.

| Decode workers | Useful output GB/s | Wall time for 12 GiB | Peak RAM GiB |
| ---: | ---: | ---: | ---: |
| 1 | 0.594 | 21.689 s | 2.015 |
| 4 | 1.815 | 7.100 s | 2.016 |
| 8 | 2.730 | 4.721 s | 2.013 |
| 16 | 4.104 | 3.139 s | 2.019 |

Each run planned 67,727 chunk uses and decoded 53,413 distinct chunks, avoiding
21.1% of repeated decodes within batches. The measured work read
9,439,401,760 encoded bytes, decoded 14,001,897,472 source bytes, and produced
12,884,901,888 output bytes. There is no persistent decoded-chunk cache.

The two 1 GiB output buffers dominate RAM. The executor's reservation grows
from 2.016 GiB with one worker to 2.255 GiB with sixteen, including conservative
codec workspace allowances. That reservation excludes metadata, plans, reader
queues, thread stacks, and allocator overhead; it is not a process RSS limit.

## CUDA comparison and uncertainty

| Pair | Baseline GB/s | Refactor GB/s |
| ---: | ---: | ---: |
| 1 | 11.834 | 11.684 |
| 2 | 12.516 | 11.716 |
| 3 | 12.409 | 11.646 |
| Median | 12.409 | 11.684 |

Median useful throughput is **5.8% lower** in the refactor runs. Baseline
throughput ranges from 11.834 to 12.516 GB/s, a spread of about 6%. With only
three pairs and uncontrolled shared NFS traffic, these results do not establish
whether the difference comes from code changes or storage variability.
Performance parity has also not been established.

The measured decode kernels take about 2.515 seconds in both versions. The
additional time appears between decode waves, with substantial gaps at batch
boundaries. A separate timing trace recorded approximately 45 ms of shared
planning and 44 ms of CUDA dispatch preparation across 35 batches including
warmup. These timings do not identify how much of the throughput difference
comes from storage waits, host preparation, or scheduling.

Correctness and retained-result lifetimes pass on CUDA. Before attributing a
performance difference to code, a follow-up should randomize the order within
pairs and collect more repetitions under controlled storage conditions. A
comparison using local storage could help isolate NFS effects.

## Checks

- CPU configuration: all 23 CTest targets pass. Its Python run passes 22 tests
  and skips 93 CUDA or optional-PyTorch cases.
- CUDA configuration on L40: all 35 CTest targets pass, including all 115
  Python tests with PyTorch installed.
- AddressSanitizer and UndefinedBehaviorSanitizer: all six selected metadata,
  planning, scheduling, and CPU pipeline targets pass.
- The CPU wheel builds and imports without CUDA. A real crop and a retained
  NumPy DLPack view work with no CUDA, cudart, or nvCOMP libraries in the
  extension dependency chain or loaded process libraries.
- Four parser fuzzers build and pass 100-run smoke checks each. The public
  Python API and new component tests pass Pyright; changed Python files pass
  Ruff. Documentation builds with MkDocs strict checking.

Correctness checks use independent expected values across raw, zstd, and Blosc
zstd encodings, supported source dtypes, missing fills, and `f32`/`bf16`
outputs. They also exercise duplicate chunk use, corrupt inputs, cache
eviction, bounded readers, memory limits, FIFO order, component reuse,
shutdown during a blocked pop, and results retained after shutdown.

The CPU/CUDA comparison exposed and fixed the CUDA decoder's handling of
Blosc's uncompressed `MEMCPYED` payload. That payload follows the 16-byte
header directly; it does not contain the compressed-block offset table.
The layout follows [C-Blosc 1.21.6](https://github.com/Blosc/c-blosc/blob/v1.21.6/blosc/blosc.c).

GDS and NUMA placement were unavailable on the L40 test host. This run does
not validate those optional paths or index queries, spatial interpolation,
NGFF level selection, or end-to-end training throughput.

## Artifacts

Scripts, complete test logs, source snapshot, scenarios, and raw benchmark
JSON are retained under
`~/tmp/2026-09-14-171630-damacy-cpu/`. CPU results are
`cpu-result-{1,4,8,16}.json`; the final CUDA comparison is in
`baseline-final-{1,2,3}.json` and `cuda-final-{1,2,3}.json`. Key validation logs
are `async-build.log`, `async-test-bench.log`, `async-ctest.log`, and
`package-finish.log`. Temporary timing instrumentation was removed from the
source and final binaries. Dataset files remain under
`~/data/damacy/throughput/`.
