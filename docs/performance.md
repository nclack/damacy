# Read performance

The [interactive report](pareto/index.html) leads with the conclusions from two
cold-read studies: random versus baked samples, sample delivery across crop sizes
at a fixed encoding, and controlled chunk/block comparisons. The explorer retains
compression Pareto plots, storage work per sample, all individual measurements,
and CSV downloads. Select an overview result to inspect its exact configuration.

The studies use an H100 node and an EPYC node, each with its own recorded storage
system. Appearance follows the system setting by default, with saved light and
dark overrides. The report retains only the two crop studies from November 6, 2026.

## Measurement definitions

| Name used throughout the report | Meaning |
| --- | --- |
| Image throughput (GiB/s) | Requested pixels × source bytes per pixel ÷ elapsed seconds ÷ 2³⁰ |
| Compression ratio (×) | Uncompressed image bytes in the stored arrays ÷ stored file bytes |
| Storage bandwidth (GiB/s) | Read bytes recorded by Linux for the storage system ÷ elapsed seconds ÷ 2³⁰ |
| IOPS (reads/s) | Completed reads recorded by Linux for that storage system ÷ elapsed seconds |
| Samples/s | Completed single-plane crops ÷ elapsed seconds |
| Storage bytes / sample (KiB) | Storage bytes ÷ completed crops ÷ 1024, calculated per execution |
| Storage reads / sample | Completed storage operations ÷ completed crops, calculated per execution |

Image throughput measures how quickly the reader delivers the requested pixels.
Storage bandwidth and IOPS measure the storage work during that time, summed over the
backing drives. They include metadata, read-ahead, and other processes using those
drives. They are kernel device counters, not counts of internal SSD flash
operations. Adjacent idle measurements remain available in **Diagnostics** and
are not subtracted. The [kernel counter definitions](https://docs.kernel.org/admin-guide/iostats.html)
describe the completed-read and sector counters used here.

All rates use wall time through completion, including the final GPU work.
Initialization, warmup, and cache clearing are excluded. One GiB is 2³⁰ bytes.
The CPU backend returns batches in system RAM; CUDA returns them in GPU memory.
CPU rates exclude a subsequent transfer to the GPU.
Converting source pixels to float32 does not increase image throughput. Stored
file lengths include padding, shard indexes, and array metadata; they are not
filesystem block allocation. Compression ratios below one mean the files are
larger than the uncompressed image bytes.

## Reader diagnostics

**Diagnostics** retains file read requests, bytes read before decompression,
bytes decoded per requested image byte, stage timings, metadata activity, and
idle storage measurements. These help explain behavior without adding more rate
names to the main plots. Stage timings overlap and cannot be added into elapsed time.

One file read request can become several storage operations; cached reads can
avoid storage operations; read-ahead can fetch extra bytes. The JSON retains both
reader and storage counters. CSV exports and configuration details retain storage
bandwidth and IOPS alongside image throughput and compression.

## Crop results

The [paired H100 crop study](pareto/index.html#study=f0166121dda806b1b509) and
[Turin CPU crop study](pareto/index.html#study=2f6063ce5114fc47763a) are complete:
**93 settings and 387 executions**, with three process repetitions per setting.
Both compare uniformly translated random crops with sequential reads of the same
baked pixels, across all seven inputs. Baked layouts use 16 shard files total per
process and one sample per compression chunk.

These are the highest observed image-throughput medians among the tested layouts
for each condition, in **GiB/s**. The selected encoding can differ between the
random and baked columns. CPU output is in system RAM; CUDA output is in GPU memory.

| Reader | Crop | Random | Baked | Baked / random |
| --- | --- | ---: | ---: | ---: |
| Turin CPU | 128² | 0.558 | 2.700 | 4.84× |
| H100 node · CPU | 128² | 0.544 | 2.349 | 4.32× |
| H100 node · CUDA | 128² | 0.533 | 2.286 | 4.29× |
| Turin CPU | 192² | 0.746 | 6.713 | 9.00× |
| H100 node · CPU | 192² | 0.713 | 6.744 | 9.46× |
| H100 node · CUDA | 192² | 0.745 | 6.129 | 8.23× |
| Turin CPU | 256² | 1.081 | 7.844 | 7.26× |
| H100 node · CPU | 256² | 1.039 | 10.784 | 10.38× |
| H100 node · CUDA | 256² | 1.021 | 9.321 | 9.13× |

**Baking is the largest improvement:** the best observed medians improve by
4.29–10.38× across these comparisons. The gain is much larger than the repeat
ranges. This includes changing the chunk shape to one stored sample per chunk,
so it measures the complete baked layout rather than isolating file order alone.

**Keep small chunks for 128² random crops.** Zstd with 16 KiB chunks has the highest
median in all three reader configurations, at 1.391× compression. On H100 CUDA it
reaches 0.533 GiB/s, versus 0.467 GiB/s for the fastest Blosc setting at 1.475×.
For 192² and 256², Blosc layouts lead the tested random choices. The 1 MiB Zstd
layout is dominated in every measured random-crop condition. Intermediate small
chunks at larger crop sizes remain unmeasured.

**Baked codec choices depend on the backend.** On the H100 node, CPU Blosc-LZ4 has
the highest baked medians at all three sizes, with about 1.42× compression.
Uncompressed data has the highest baked CUDA medians; at 128² its advantage over
Blosc-LZ4 is under 1%, so both remain useful candidates. At 256², CUDA Blosc-Zstd
with 16 KiB requested blocks reaches **7.173 GiB/s at 1.463× compression**.
Whole-chunk blocks reach **2.726 GiB/s at 1.473×**. That small compression gain
comes with a large throughput cost, so block size remains an important parameter.

**There is no consistent Turin advantage in this sweep.** The best random-crop medians are
close across the machines. For baked 256² Blosc-LZ4, the H100 node's CPU reaches
10.784 GiB/s versus 7.844 GiB/s on Turin, with separated repeat ranges. These are
different hosts and storage arrays. CPU/GPU training comparisons also need to
include the CPU-to-GPU transfer and the actual training loader.

The overview keeps one encoding recipe per line when comparing samples/s across
crop sizes. The following best observed medians are separate selections, not a
fixed-recipe curve: H100 CUDA's best
random medians are about **16,289, 10,127, and 7,804 samples/s** at 128², 192², and
256²; its best baked medians are **69,914, 83,309, and 71,263 samples/s**. More image
bytes per second can accompany fewer completed samples. The
[crop protocol](#random-crops-and-baked-samples) also explains the fixed CPU
input-buffer limit and the extra read/decode groups at 128².

Lower IOPS can accompany faster sample delivery. Holding the codec uncompressed
at 128² on H100 CUDA, random crops use about **420.9 kB of storage reads and 3.217
storage operations per sample**, versus **36.1 kB and 0.275 operations** for baked
samples. Image throughput rises from 0.366 to 2.286 GiB/s. This compares complete
layouts, including different chunk shapes and file counts.

The storage view divides each execution's bytes and operations by its completed
samples, then summarizes the resulting values. It does not divide independently
aggregated rate medians. Operations per sample can be below one because one
storage operation can serve multiple samples. The table and sidebar retain
storage bandwidth and IOPS for inspecting the measured rate.

The paired H100 run contains **62 settings and 258 executions**. It used one H100
80 GB GPU, an Intel Xeon Platinum 8462Y+, 32 allowed logical CPUs, 16 CPU decode
workers, and 16 file workers. It finished in **45 minutes 16 seconds**, including
all 42 native tests, the reader check, staging, and measurements. The Turin run
contains **31 settings and 129 executions**, on an AMD EPYC 9655P with the same
worker counts and allowed CPU count. Each machine used XFS on its own array of
eight 7.68 TB Intel NVMe drives. Exact hosts and inventories are recorded in the
report's Storage system section.

All **1,353 timed passes** verified zero resident shard-data pages: 448 on Turin
and 905 on H100. Peak overlapping reads reached 16 in every execution. H100 IOPS
ranged from 11,447 to 126,002, with adjacent idle reads at most 70 IOPS and 0.36%
of the corresponding measured rate. Turin IOPS ranged from 12,514 to 128,561;
idle reads were at most 5 IOPS and 0.012%.

H100's median min–max repeat spread was **1.74%**, with a maximum of 9.07%; its
bracketing control spreads were 0.96–7.84%. Turin was noisier: median 11.34%,
maximum 24.61%, and control spreads of 11.51–26.33%. These are observed ranges
from three repetitions and six controls per condition, not confidence intervals.
The plotted frontiers use medians, so small apparent advantages should not be
treated as established dominance.

The [H100 JSON](pareto/data/f0166121dda806b1b509.json) and
[Turin JSON](pareto/data/2f6063ce5114fc47763a.json) retain every observation, source
geometry, query, and source/binary hash. Paired query streams are downloadable
from each result. The protocol below describes the separate generative, image,
and video training follow-ups; these measurements cover single-plane storage
workloads.

## Random crops and baked samples

The current [`bench/studies/microscopy.json`](https://github.com/nclack/damacy/blob/bench/microscopy-pareto/bench/studies/microscopy.json)
defines the smaller study whose [measured results are above](#crop-results).
It compares 128 × 128, 192 × 192, and 256 × 256 crops from the complete seven-input
mixture.

**Random crops** choose each origin uniformly from every valid translation in the
expanded source volume. The entire crop must be in bounds. Crops can cross both
chunk and shard boundaries. Repeated and overlapping coordinates are allowed;
there is no spatial-tile quota or exclusion of previously visited chunks.

**Baked samples** contain exactly those same pixels, in the original source type.
Each compression chunk holds one sample. Within each source, sample `i` goes to
file `i % source_file_count`, at offset `i // source_file_count`. Reading the paired
query sequence therefore advances through each file. Preparation checks exact
pixel round trips and verifies that successive chunks occupy contiguous byte
ranges in each shard file. Both query streams are retained, including the source
coordinates used to bake the timed samples.

There are **16 baked shard files total per process**, across all sources. Native
types remain in separate arrays: the five uint16 inputs get two files each,
DynaCell float32 gets five, and COSEM uint8 gets one. Files are approximately
1 GiB uncompressed; exact shapes and encoded sizes are recorded per source.
The baked corpus contains equal sample counts per input, rounded to whole files
and batches. A timed pass reads forward through a prefix of every file. It does
not scan a whole epoch. The random layout retains the earlier 2048² volumes and
512² spatial shards, with 16 files per source (112 in total).

The initial layout choices are:

| Encoding | Random layout | Baked layout |
| --- | --- | --- |
| Uncompressed control | 256 KiB chunks | One sample per chunk |
| Zstd | 1 MiB chunks | One sample per chunk |
| Blosc-Zstd, 16 KiB requested blocks | 256 KiB chunks | One sample per chunk |
| Blosc-Zstd, whole-chunk blocks | 256 KiB chunks | One sample per chunk |
| Blosc-LZ4, whole-chunk blocks | 256 KiB chunks | One sample per chunk |

The compressed random layouts cover the observed all-source frontier choices
from the earlier Turin CPU, H100 CPU, and H100 CUDA measurements. A **16 KiB Zstd
check at 128²** keeps small chunks in consideration. Old dominated layouts are not
assumed to remain dominated after changing the workload. Baked chunks range from
16 KiB for a 128² uint8 sample to 256 KiB for a 256² float32 sample; a nominal
256 KiB label would be misleading. Compression level remains three, with bitshuffle
for Blosc. Actual block sizes are verified from stored headers.

Batches contain **36 MiB of float32 output**, and lookahead contains **144 MiB** at
every crop size. That gives batches of 576, 256, and 144 samples for 128², 192², and
256² respectively. Passes return at least 1 GiB per source, rounded to complete
batches. Worker counts and CPU/GPU memory limits stay fixed. Each condition has its
own uncompressed controls before and after the settings, and three process
repetitions by default. The pilot changes repetitions and duration, not the source
mixture or layout selection.

The CPU reader holds at most 256 compression chunks in each input buffer.
With one chunk per baked sample, 128² batches need three read/decode groups;
192² and 256² batches each fit in one. The crop curves include this batching
effect and per-sample processing costs.

This is **62 settings and 258 executions** across CPU and CUDA, with one all-source
mixture. Leave-one-out runs, extra repeats, intermediate chunk sizes, and grouping
multiple baked samples per chunk are follow-ups around the resulting frontiers.
The definition can enable dataset omissions while retaining the remaining query
coordinates and order; omissions also reduce the baked file count. They are not
part of the first sweep.

Each crop size, access pattern, backend, and machine has its own Pareto frontier. The **Crop size and
samples/s** view compares completed crops per second at the fixed byte budgets.
It converts the image throughput using the exact source composition of each query
stream. A line follows one encoding recipe, not a frontier across different tasks.
Compression ratio still describes the actual stored arrays: random arrays have
equal logical bytes per source; baked arrays have equal sample counts. Their
compression ratios therefore weight native source types differently. The paired
read rates use the same source samples and byte counts.

The cold-pass protocol below applies to both layouts. Every timed pass begins with zero
resident shard-data pages, following warmup or preparation. Cache reuse and
read-ahead within a pass are part of the measured reader. This is not a guarantee
that every individual crop reaches storage.

### Both Katamari training paths

The spatial sweep is a storage experiment shared by both targets. It is not a
replay of either full training loader. The following source-read contracts were
checked against Katamari revision
[`41ca4527e420`](https://github.com/evolutionaryscale/evolutionaryscale/tree/41ca4527e42017ad60c8e948fbb3c087168c294e/projects/katamari):

| Follow-up profile | Source reads to preserve |
| --- | --- |
| Generative ROI | One timepoint and Z plane; all native channels before output channel selection; random or centroid-centered ROI before normalization and resizing |
| Image pretraining | One timepoint and Z plane; all manifest-selected channels; usually 256² source crops, with a 224² exception |
| Video pretraining | 12 or 16 timepoints, one Z plane, all manifest-selected channels; temporal stride 1 or 2 |

The [generative v7 configuration](https://github.com/evolutionaryscale/evolutionaryscale/blob/41ca4527e42017ad60c8e948fbb3c087168c294e/projects/katamari/configs/experiment/generative/rae/fm/baselines/v7.yaml#L263)
produces 256² model inputs, but nominal source ROIs include 96² and 160², derived
from physical field of view and pixel size. Manifest calibration can override
those defaults. The chosen 128/192/256 storage sweep does not stand in for that
actual ROI distribution. The active
[image/video recipe](https://github.com/evolutionaryscale/evolutionaryscale/blob/41ca4527e42017ad60c8e948fbb3c087168c294e/projects/katamari/configs/experiment/prod/img-vid/active/vit_s.yaml#L13)
uses the legacy loader with homogeneous batches and log-weighted source selection.
Its model channel limits are applied after source reads. Reading only the final
channel count would understate storage work.

A subsequent trace-based check should replay these three profiles on the selected
layouts, retaining actual source/channel/time/ROI selections and source weights.
For generative baking, preserve captions, conditioning, channel identities, masks,
calibration, and paired reference geometry as well as pixels. For video, retain
real frame relationships and temporal stride. The tiled distribution corpus does
not supply realistic temporal or cross-channel correlations, so stacking its
planes would not establish video-training performance. Keep these profile checks
separate rather than multiplying every time, channel, crop, and codec choice.

### Reasoning about eight GPUs

Only one GPU is needed for this study. With one process per GPU and disjoint file
ownership, 16 active files per process means **128 active files per eight-GPU node**.
Sixteen is a starting point per process, not a worker multiplier or an established
scaling optimum. The processes share storage, CPU, and memory bandwidth; neither
throughput nor file-worker counts should be extrapolated linearly.

The measured H100 node has 128 logical CPUs, of which this one-GPU allocation
uses 32. Dividing that node evenly among eight GPU processes would give 16 logical
CPUs per process. The measured CPU budget therefore also needs reconsideration
when scaling.

For a target of `r` samples/s per GPU and `b` measured storage bytes per sample,
the node needs approximately `8 × r × b` storage bytes/s. Apply the same calculation
to storage operations per sample. Compare these demands with measured node limits
and retain headroom for decoding and other work. Later scaling tests should vary
files and outstanding work per process while enforcing disjoint samples and file
ownership. Generative source synchronization and image/video batch partitioning
must also retain their original source weights.

## Source distributions and cold reads

The version-3 [corpus manifest](https://github.com/nclack/chucky-benchmarks-data)
provides seven inputs: OpenCell DNA and protein, BBBC010 brightfield, JUMP-Scope
fluorescence, DynaCell phase, COSEM EM, and BBBC022 MitoTracker. Original source
URLs, licenses, hashes, shapes, and per-file sizes are retained in each result.

Each source becomes a 2048 × 2048 frame sequence. A 4 × 4 spatial grid gives
16 shard files per frame, with each shard containing **1 GiB of uncompressed
pixels**. Depth depends on the original type; shapes below are `[plane, Y, X]`.
Chunk sizes also count uncompressed pixel bytes.

| Source type | Expanded volume | Shard shape | 1 MiB chunk shape | Logical volume bytes |
| --- | --- | --- | --- | --- |
| uint8 | 4096 × 2048 × 2048 | 4096 × 512 × 512 | 4 × 512 × 512 | 16 GiB |
| uint16 | 2048 × 2048 × 2048 | 2048 × 512 × 512 | 2 × 512 × 512 | 16 GiB |
| float32 | 1024 × 2048 × 2048 | 1024 × 512 × 512 | 1 × 512 × 512 | 16 GiB |

At 1 MiB, each chunk spans the shard's full 512 × 512 XY footprint, and each
1 GiB shard contains 1024 chunks. COSEM uses uint8, DynaCell uses float32, and
the other five inputs use uint16.

Compressed shard file sizes vary by source and codec and are retained individually.
The writer processes and verifies one complete shard at a time. Original values,
types, checksums, and attribution are retained. Source planes repeat cyclically;
each output plane uses a seeded spatial offset and periodic tiling. Adjacent
spatial shards agree at their boundaries. This creates distinct stored extents
with the source's pixel distribution and local spatial texture. It does not
create new specimens or establish realistic depth correlations, especially for
single-plane inputs. Chunk layouts extending across planes must be interpreted
with that limitation.

Every timed pass evicts and checks every shard-data file before running many
batches through the pipeline. It records storage counters around the pass and
repeats checked passes until the minimum time is reached. Pixels are not preloaded
into RAM for measurement. Codecs and metadata are warmed, with eviction after
preparation and warmup. Kernel read-ahead and overlapping reads within a pass are
part of the measured buffered reader. SSD/controller caches are not controlled.

The 16 file workers and 16 CPU decode workers are resource limits. They do not
mean that 16 different files are always being read at once. The report records
planned distinct files per batch and lookahead window separately from the measured
peak number of overlapping file read calls, which can include calls to the same file.

## Reproduce a study

Use a materialized version-3 corpus with `manifest.json` and the declared raw
files. Planning reads metadata only. The default definition is the crop study:

```sh
uv run --script bench/pareto.py plan bench/studies/microscopy.json \
  --corpus /path/to/microscopy --output /shared/study/plan.json
```

Use `--pilot` for two shorter repetitions instead of three; the new crop matrix
and complete source mixture are unchanged. Dataset omissions are deferred.

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

Preparation can reuse earlier arrays with `prepare --reuse /shared/previous/data`.
It checks the exact layout and every file checksum before reuse. Reused arrays are
linked in the preparation directory; copy the prepared file inventory to local
storage before measuring, as `bench/jobs/pareto-mixed.sh` does. Raw layouts in the
new definition retain their earlier identities, so only the baked arrays need encoding.

## Preview a pull request

The docs workflow builds and checks the page in Chromium. Pull requests from a
branch in this repository publish their site under
`https://nclack.github.io/damacy/pr-preview/pr-N/`, with the report at `pareto/`.
The workflow summary links to the preview. The production report keeps its existing URL. Every docs build also uploads a `docs-site` artifact;
download and unzip it, then run `python -m http.server` from that directory.
Fork pull requests provide the artifact without publishing to Pages.
