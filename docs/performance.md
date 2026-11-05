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

## Mixed-source cold NVMe results

The [November 5 CPU study](pareto/index.html#study=542501404495a92083a9)
contains **126 settings and 308 executions**, using all seven microscopy assets,
nine layouts, two read patterns, and six dataset omissions. Each setting has two
process repetitions. It ran on an AMD EPYC 9655P with 32 allowed logical CPUs,
16 decode workers, 16 file workers, and eight 7.68 TB Intel PCIe NVMe drives.
The prepared files total 769.17 GiB, representing 1008 GiB of uncompressed pixels.

For the complete source mixture, two layouts form the observed frontier in both
read patterns:

| Layout | Compression | Crop useful GiB/s | Whole-frame useful GiB/s |
| --- | ---: | ---: | ---: |
| Zstd, 1 MiB chunks | 1.441× | 1.205 | 5.579 |
| Blosc-Zstd, 256 KiB chunks and blocks | 1.504× | 1.030 | 4.472 |

These are medians from two repetitions, not an established ordering of close
settings. Across the study, the median min–max throughput spread was 2.25% of
the median, with a maximum of 16.82%. Bracketing raw-control spreads ranged from
1.13% to 5.64%. The plots retain individual ranges and the controls for each
source mixture and read pattern.

All 758 measured passes passed the zero-resident-page check and recorded NVMe
reads. Peak overlapping application read calls reached 16 in every execution.
Device read rates ranged from about 32,000 to 141,000 IOPS across executions;
these rates describe the measured requests and include device activity outside
the process. The adjacent idle rate reached 455 read IOPS. For all-source
whole-frame Zstd/1 MiB reads, the median device rate was 70,375 IOPS and
7.343 GiB/s, while useful source throughput was 5.579 GiB/s and returned
float32 throughput was 10.414 GiB/s.

Omitting DynaCell changes the highest-median layout to Blosc-Zstd/256 KiB for
both read patterns. Omitting BBBC010 changes whole-frame Zstd/1 MiB throughput
by only −0.47%, while its compression falls 10.14%, from 1.441× to 1.295×.
The sensitivity panel shows both axes and each mixture's raw-normalized speedup.
Omissions change source types and file counts as well as the pixel distribution;
see the [study protocol](#mixed-source-microscopy-study) for those limits.
The [retained JSON](pareto/data/542501404495a92083a9.json) includes every observation,
cache check, device counter, and source, query, and binary hash.

## Earlier cold NVMe pilot results

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

## Earlier single-batch cold protocol

The retained cold pilot used `"cache_policy": "cold"`. The cached pilot
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

## Mixed-source microscopy study

The updated [`bench/studies/microscopy.json`](https://github.com/nclack/damacy/blob/main/bench/studies/microscopy.json)
uses every asset in the version-3 [corpus manifest](https://github.com/nclack/chucky-benchmarks-data).
The current seven inputs are OpenCell DNA and protein, BBBC010 brightfield,
JUMP-Scope fluorescence, DynaCell phase, COSEM EM, and BBBC022 MitoTracker.
The November 5 mixed-source CPU results above use this geometry and protocol.
The older single-input pilots retain their original, smaller inputs.

Each source becomes a 2048 × 2048 frame sequence. A 4 × 4 spatial grid gives
16 shard files per frame, with each shard containing **1 GiB of uncompressed
pixels**. Depth depends on the original type; shapes below are `[plane, Y, X]`.

| Source type | Expanded volume | Shard shape | Logical volume bytes |
| --- | --- | --- | --- |
| uint8 | 4096 × 2048 × 2048 | 4096 × 512 × 512 | 16 GiB |
| uint16 | 2048 × 2048 × 2048 | 2048 × 512 × 512 | 16 GiB |
| float32 | 1024 × 2048 × 2048 | 1024 × 512 × 512 | 16 GiB |

Compressed shard file sizes vary by source and codec and are retained individually.
The writer processes and verifies one complete shard at a time. Original values,
types, checksums, and attribution are retained. Source planes repeat cyclically;
each output plane uses a seeded spatial offset and periodic tiling. Adjacent
spatial shards agree at their boundaries. This creates distinct stored extents
with the source's pixel distribution and local spatial texture. It does not
create new specimens or establish realistic depth correlations, especially for
single-plane inputs. Chunk layouts extending across planes must be interpreted
with that limitation.

The default matrix compares raw bytes, Zstd, Blosc-Zstd, and Blosc-LZ4; compression
level is three and Blosc uses bitshuffle. Chunk sizes are 16, 64, 256, and 1024 KiB;
Blosc block requests are 16 KiB or the whole chunk. Chunk dimensions divide the
512 × 512 spatial shards. When a large chunk cannot fit in one spatial plane, it
extends along the plane axis. Actual block sizes are read back from encoded headers.
The [Zarr sharding specification](https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/)
requires each inner chunk dimension to divide its shard dimension.

Two workloads use retained, deterministic query streams:

- **Shard-balanced 256 × 256 crops.** Each source contributes equally many
  samples. Queries rotate through all 16 spatial shards, choose a random crop
  inside each selected shard, and avoid revisiting any maximum-chunk-size plane
  extent in that shard during a pass. Crops may cross inner chunk boundaries,
  but do not cross shard boundaries; this is an explicit workload constraint.
- **Whole-frame reads.** Each source contributes sampled 2048 × 2048 planes,
  each touching all 16 spatial shards. Selected planes do not reuse the same
  maximum-chunk-size depth extent within a pass.

Each pass returns 1 GiB of float32 samples per input: 4096 crops or 64 whole
frames, mixed across the seven sources. The complete mixture therefore returns
7 GiB and requests 3.75 GiB of source pixels per pass. Original dtype sizes are
used when computing useful bytes. Compression aggregates the equal 16 GiB
source volumes by summing logical and stored bytes; read weights are equal
sample counts, which give unequal source-byte contributions across types.

Leave-one-out cases remove a whole dataset collection and retain all remaining
coordinates and their relative ordering. Both OpenCell channels are removed
together. The pass contains less work after an omission, with the same work per
retained input. The report compares useful and returned rates and changes in
speedup against each mixture's uncompressed reference. Changes also reflect the
reduced number of source files; they are sensitivity measurements, not causal
estimates of a source's compression cost.

The new `cold-pass` policy evicts and checks every shard-data file before a
multi-batch pass, then lets the pipeline overlap reads and decoding through the
whole pass. It records device counters around each pass and repeats checked
passes until the minimum time is reached. It does not preload pixels into RAM
for measurement. Codecs and metadata are warmed; eviction follows any preparation
or warmup reads. Kernel readahead and incidental reuse within a pass remain part
of the measured buffered reader. The original `cold` policy remains available
for reproducing the earlier one-batch pilots.

The study uses 16 file workers, 16 CPU decode workers, and a lookahead of 512
samples. These are resource limits, not a claim that 16 shard files are always
being read simultaneously. The report records planned distinct shard files per
batch and lookahead window separately from the measured peak number of overlapping
file read calls. The latter can include multiple calls to the same file and
excludes the metadata reader's separate activity.

Both the pilot and full study include all seven assets, both backends, both
workloads, the complete mixture, and six dataset omissions. The pilot has nine
layouts per input (63 encoded arrays, 1008 GiB logical preparation), 252 settings,
and 616 process executions across both backends. Its minimum timed work totals
1456 seconds; encoding, validation, startup, warmup, cache clearing, and completed
pass overruns are additional. The full study uses 22 layouts per input, 616
settings, and 2016 executions. It encodes 2464 GiB of logical pixels. Encoded
storage requirements are measured during preparation, rather than estimated
from the small earlier pilots.

Repeated results report medians and observed min–max ranges. Bracketing raw
controls show variation during each condition. The previous definitions are
retained in `bench/studies/microscopy-pilot-v1.json` for reproducing old data.


## Reproduce a study

Use a materialized version-3 corpus with `manifest.json` and the declared raw
files. Planning reads metadata only:

```sh
uv run --script bench/pareto.py plan bench/studies/microscopy.json \
  --corpus /path/to/microscopy --pilot --output /shared/study/plan.json
```

Omit `--pilot` for all 22 layouts per input. Both modes include every source and all six dataset omissions.

Preparation and measurement belong on compute nodes. Build `damacy_bench` with
CUDA enabled to measure both backends. For CPU-only measurements without a GPU
driver, configure with `-DDAMACY_CUDA=OFF`. Preparation checks input hashes, encodes
each layout, verifies every decoded pixel, and records the stored files and
Blosc headers:

```sh
uv run --script bench/pareto.py prepare /shared/study/plan.json \
  --corpus /path/to/microscopy --output /node-local/study/data --workers 3
uv run --script bench/pareto.py run /shared/study/plan.json \
  --data /node-local/study/data --output /shared/study/results \
  --storage 'Local NVMe; checked cold multi-batch passes'
uv run --script bench/pareto.py report /shared/study/results \
  --output docs/pareto/data
```

`--workers` selects how many layouts to prepare at once; the default is one.
Completed layouts can be reused with a different worker count. Every stored file
is checked again after copying and before measurements; those checks are untimed.

Prepare on verified node-local NVMe, or copy prepared files there before running. `--backend cpu`
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
