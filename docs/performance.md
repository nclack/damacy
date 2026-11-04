# Read performance

The [interactive Pareto report](pareto/index.html) compares useful read throughput
with compression. Each point is a measured chunk, codec, and Blosc block setting.
CPU and CUDA, image inputs, read patterns, machine sessions, and storage conditions
have separate frontiers. Additional plots show application read requests and
NVMe device IOPS against bandwidth. The appearance selector defaults to the
system setting and remembers explicit light or dark choices.

The existing [throughput history](https://nclack.github.io/damacy/throughput/) and
[timing history](https://nclack.github.io/damacy/timings/) retain the daily CI
measurements. Those fixed scenarios help track changes on the CI machine.
The retained Pareto studies explore which settings work well on a recorded machine.

## What the axes count

| Measurement | Definition |
| --- | --- |
| Useful GiB/s | Requested pixels × source bytes per pixel ÷ elapsed seconds ÷ 2³⁰ |
| Compression fold | Original image bytes ÷ stored Zarr file bytes |
| Application reads/s | Completed data read requests after combining adjacent requests ÷ elapsed seconds |
| Encoded GiB/s | Bytes in those application read requests ÷ elapsed seconds ÷ 2³⁰ |
| Mean application read | Encoded read bytes ÷ application read requests |
| Requests / useful GiB | Application read requests ÷ useful source bytes × 2³⁰ |
| NVMe read IOPS | Device read completions during measured batches ÷ elapsed seconds |
| Device GiB/s | Device sectors read × 512 ÷ elapsed seconds ÷ 2³⁰ |
| Read amplification | Encoded bytes read by the pipeline ÷ useful source bytes returned |
| Decode amplification | Bytes reported by the decode stage ÷ useful source bytes returned |

Stored file bytes include edge padding, shard indexes, and array metadata. They
are file lengths, not filesystem block allocation. Uncompressed settings can
therefore have compression folds below one. A missing decode-byte counter is
shown as unavailable. Overlapping or repeated requests can share encoded reads,
so read amplification can be below one even without compression.

Returned tensor throughput is available in configuration details. For uint16
input converted to float32, it is twice the useful source-byte rate. The primary
axis keeps this conversion from inflating the reported read speed.

Elapsed measurement time includes pushing queries, reading, decoding, conversion,
assembly, and waiting for the last CUDA work to finish. Setup and warmup are
separate. Cold measurements also exclude cache clearing and counter collection.
Stage intervals overlap and cannot be summed into elapsed time.

Application requests use the native `reads_issued` counter, not I/O wave counts
or accumulated stage time. They exclude metadata reads and are not counts of
NFS requests or physical disk commands. The kernel can split or combine requests,
serve them from RAM, or read ahead. Device counters are reported separately and
are shared with other activity on those devices. Their adjacent idle rates are
shown without subtracting an assumed background rate. A high IOPS value alone
does not make a setting better; request size and useful throughput provide context.

## Cold NVMe pilot results

The [cold CPU pilot](pareto/index.html#study=5add8a3675356a860c8e) contains
72 executions of 24 settings on an AMD EPYC 9655P, with eight allowed logical
CPUs and an array of eight 7.68 TB Intel PCIe NVMe drives. It covers BBBC010,
DynaCell, and COSEM, both read patterns, and four layouts per input.

All 24,825 measured batches passed the zero-resident-page check and recorded
device reads. Device counters totaled 99.77 GiB read during the measured batches.
The adjacent idle intervals recorded no device reads, but did record writes;
the drives were not reserved exclusively for this study. The median throughput
range across repeats was 3.0% of the median, with a maximum of 9.9%.
Bracketing reference ranges also reached 9.9%. The
[retained JSON](pareto/data/5add8a3675356a860c8e.json) includes cache checks,
device counters, metadata activity, source hashes, and individual observations.
Its metadata read-operation counts distinguish startup from timed work. The
cache-miss fields in that retained native result are snapshots, so the page does
not interpret them as additional timed metadata reads.

This validates the cold protocol on local storage. It is a CPU-only pilot on a
different machine from the cached H100 study; their rates are not a paired
comparison of cache state. A cold CUDA study and the full layout matrix remain
unmeasured.

## Cold NVMe protocol

The study definition now selects `"cache_policy": "cold"`. The earlier pilot
below used preloaded files; it remains a separate cached study and is not
evidence of cold NVMe performance.

1. Copy prepared files to a private directory on the compute node. Resolve the
   mounted XFS/ext4 block device through partitions and storage arrays to its
   physical devices. Require PCIe NVMe, record device models, and reject network
   mounts or storage whose backing devices cannot be verified.
2. Warm the reader, codecs, and GPU. Retain startup metadata-operation counts,
   including array metadata and shard-index cache misses, then reset measurement
   counters. The native result records startup and measured metadata separately.
3. Flush writes to the owned data files, then use `POSIX_FADV_DONTNEED` and
   `mincore` to evict and check every data-file page before **each** measured batch.
   A successful advisory call is insufficient: any remaining resident page fails
   the run. This does not flush the whole node's cache or clear the reader's
   metadata cache. The kernel documents both the advisory nature of
   [file-cache eviction](https://man7.org/linux/man-pages/man2/posix_fadvise.2.html)
   and the snapshot nature of [residency checks](https://man7.org/linux/man-pages/man2/mincore.2.html).
4. Time one batch through completion, including the final CUDA work. Repeat cold
   batches until their summed elapsed time reaches the requested window. Cache
   clearing, residency checks, initialization, and idle controls stay outside that
   sum. The same query seed is used across layouts and backends. Whole-plane
   batches contain at most the input's number of planes.
5. Record NVMe read completions, bytes, and writes around each timed batch. Reject
   a batch with no device reads. Retain 100 ms idle intervals before and after the
   execution to expose background traffic. Device statistics follow the
   [kernel counter definitions](https://docs.kernel.org/admin-guide/iostats.html).

This measures **Linux page-cache-cold batches with warmed codecs and metadata**.
SSD/controller caches are not controlled. Repeated small image packs do not
establish sustained streaming bandwidth, and overlapping requests within a batch
can share reads. Cold and cached runs have separate measurement conditions and
frontiers. For a sustained-storage study, use a larger collection of distinct
file extents and vary batch size and outstanding work; do not interpret this
pilot as the drive's maximum bandwidth or IOPS.

## Cached H100 pilot results

The [November 4, 2026 pilot](pareto/index.html#study=f0aa215f44b8b59093a0)
contains 144 executions of 48 settings on an NVIDIA H100 80 GB and Intel Xeon
Platinum 8462Y+, with 16 allowed logical CPUs. It uses BBBC010, DynaCell, and
COSEM inputs, two process repetitions, eight CPU decode workers, and eight file
workers. Files were copied to node-local XFS storage and preloaded into the page
cache. The [study JSON](pareto/data/f0aa215f44b8b59093a0.json) retains individual
measurements, the exact allocation, and source and binary hashes.

The block-size tradeoff is already visible. For DynaCell whole-plane CUDA reads,
with 256 KiB chunks and otherwise identical Blosc-Zstd settings:

| Blosc block size | Useful GiB/s, median | Observed range | Compression fold |
| --- | ---: | ---: | ---: |
| 16 KiB | 11.67 | 11.48–11.85 | 1.137× |
| 256 KiB | 1.102 | 1.102–1.103 | 1.151× |

The larger blocks reduce stored bytes by about 1.2% while reducing throughput
by about 10.6-fold in this condition. These small, replayed image packs include
overlapping and repeated requests; the rates describe those requests, not
streaming through an equally large collection of distinct images.

Across the 48 settings, the median min–max throughput spread was 1.9% of the
median; the largest was 15.9%. Uncompressed medians from the 0.5-second windows
were within 5.5% of the medians from the three-second reference windows.
Reference ranges reached 9.4%. Small frontier differences need more repetitions
and longer measurements before being treated as a reliable ordering.

The measurement runner took 6.1 minutes, including 3.2 minutes of timed work.
Applying its average process overhead to the cached full matrix gives approximately
79 minutes before preparation and validation. Cold runs also spend time clearing
and checking file pages, so that estimate does not apply to the cold protocol.
The full matrix has not been measured.

## Microscopy study

[`bench/studies/microscopy.json`](https://github.com/nclack/damacy/blob/main/bench/studies/microscopy.json)
defines the matrix. It uses version 3 of the public
[microscopy corpus](https://github.com/nclack/chucky-benchmarks-data), also used by
the [chucky microscopy analysis](https://acquire-project.github.io/chucky/microscopy.html).
Inputs retain their original pixel values, shape, type, checksum, and attribution.

The seven inputs cover OpenCell DNA and protein, BBBC010 brightfield, JUMP-Scope
fluorescence, DynaCell phase, COSEM EM, and BBBC022 MitoTracker. These are small
image packs, not a representative sample of every microscopy workload.

The study compares:

- Uncompressed bytes, Zstd level 3, and Blosc Zstd level 3 with bitshuffle.
- 16, 64, 256, and 1024 KiB chunks, with a plane extent of one. The remaining
  dimensions are powers of two, with the longer extent along X.
- Blosc block requests of 16 KiB and the whole chunk. Duplicate requests are
  removed, and actual block sizes are read from the stored headers.
- Random 256 × 256 crops and sequential whole-plane reads.
- CPU and CUDA decoding, with the same encoded files and queries.

Shard shape stays at `[1, 1024, 1024]`; CPU decode and file worker counts stay
at eight. The reader allows 64 Blosc blocks per chunk to cover the full matrix.
Both backends return float32. These are controlled comparisons at
recorded resource limits, not claims about the best possible CPU worker count
or shard geometry.

Cold executions use the protocol above. An explicit `"preloaded files"` cache
policy instead preloads encoded files and replays fixed query passes until both
the work budget and minimum duration are satisfied. The `bytes_per_pass` profile
field applies to that cached mode; cold mode always uses one batch per pass.

The full matrix has 105 encoded layouts and 420 measurement settings. Three
randomized rounds plus bracketing uncompressed controls make 1,428 executions.
The pilot covers three inputs spanning uint8, uint16, and float32, four layouts
per input, both backends, and both read patterns: 48 settings and 144 executions.
Its requested measurement windows total 192 seconds before preparation, process
startup, warmup, and work-budget overruns.

Pilot results are labelled. Repeats report medians and observed min–max ranges;
they are not confidence intervals. Controls before and after each condition
show reference variation. The full study should follow a pilot review of
runtime, repeatability, and agreement between the short and longer control windows.

## Reproduce a study

Use a materialized version-3 corpus with `manifest.json` and the declared raw
files. Planning reads metadata only:

```sh
uv run --script bench/pareto.py plan bench/studies/microscopy.json \
  --corpus /path/to/microscopy --pilot --output /shared/study/plan.json
```

Omit `--pilot` to plan all seven inputs and 105 layouts.

Preparation and measurement belong on compute nodes. Build `damacy_bench` with
CUDA enabled to measure both backends. Preparation checks input hashes, encodes
each layout, verifies every decoded pixel, and records the stored files and
Blosc headers:

```sh
uv run --script bench/pareto.py prepare /shared/study/plan.json \
  --corpus /path/to/microscopy --output /shared/study/data
uv run --script bench/pareto.py run /shared/study/plan.json \
  --data /node-local/study/data --output /shared/study/results \
  --storage 'Local NVMe; file cache checked cold before every batch'
uv run --script bench/pareto.py report /shared/study/results \
  --output docs/pareto/data
```

Copy the prepared data to the node-local path before running. `--backend cpu`
selects only CPU measurements. Each process has a timeout; failures retain
stdout, stderr, and its scenario. Repeating the same command resumes completed
observations only when the plan, files, source, binary, machine, allocation,
and resource settings still match. A new session needs a new result directory.

The public export retains raw observations and file hashes and removes local
input paths. It rejects incomplete studies and mismatched observations. Serve
the report over HTTP, or build the documentation site; the page loads its study
JSON separately. The daily CI workflow does not start these retained studies.

## Preview a pull request

The docs workflow builds and checks the page in Chromium. Pull requests from a
branch in this repository publish their site under
`https://nclack.github.io/damacy/pr-preview/pr-N/`, with the report at `pareto/`.
The workflow summary links to the preview. Production docs and benchmark histories
keep their existing URLs. Every docs build also uploads a `docs-site` artifact;
download and unzip it, then run `python -m http.server` from that directory.
Fork pull requests provide the artifact without publishing to Pages.
