# Read performance

The [interactive Pareto report](pareto/index.html) compares **Image throughput
and compression**, **Storage reads**, and, for crop studies, **Crop size and
samples/s**. Each point is a measured chunk, codec, and Blosc block setting.
CPU and CUDA, source mixtures, read patterns, crop sizes, and machine sessions
have separate frontiers. All panels within a study use one recorded
storage system. Its host, processor, GPU, filesystem, and drives appear once in
**Storage system**. Appearance follows the system setting by default, with saved
light and dark overrides.

The existing [throughput history](https://nclack.github.io/damacy/throughput/) and
[timing history](https://nclack.github.io/damacy/timings/) retain the daily CI
measurements. Those fixed scenarios track changes on the CI machine.

## Four measurements

| Name used throughout the report | Meaning |
| --- | --- |
| Image throughput (GiB/s) | Requested pixels × source bytes per pixel ÷ elapsed seconds ÷ 2³⁰ |
| Compression ratio (×) | Uncompressed image bytes in the stored arrays ÷ stored file bytes |
| Storage bandwidth (GiB/s) | Read bytes recorded by Linux for the storage system ÷ elapsed seconds ÷ 2³⁰ |
| IOPS (reads/s) | Completed reads recorded by Linux for that storage system ÷ elapsed seconds |

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

The old **Encoded read GiB/s** label counted the bytes in Damacy's file read
requests before decompression. These are compressed bytes when a codec is used
and raw bytes for uncompressed files. It did not measure bytes at the drives.
The old **Application read requests/s** counted those byte-range requests after
adjacent requests were combined; it did not measure storage IOPS.

For example, one 1 MiB file request could become eight 128 KiB storage reads.
That would count as one reader request and eight storage operations, even if
both byte counts were 1 MiB. Cache hits can avoid storage reads; read-ahead can
fetch extra bytes. Thus both request counts and byte counts can differ between
the reader and the storage system.

The separate application-read plot has been removed. **Diagnostics** retains
file read requests/s, file read bandwidth, mean request size, bytes read or decoded
per requested image byte, stage timings, metadata activity, and idle storage
measurements. These help investigate the reader. Stage timings overlap and
cannot be added into elapsed time. The JSON retains all original counters,
including output-byte rates after float32 conversion. CSV exports use the same
four main measurement names as the plots and sidebar.

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

**There is no consistent Turin advantage in this sweep.** Random-crop medians are
close across the machines. For baked 256² Blosc-LZ4, the H100 node's CPU reaches
10.784 GiB/s versus 7.844 GiB/s on Turin, with separated repeat ranges. These are
different hosts and storage arrays. CPU/GPU training comparisons also need to
include the CPU-to-GPU transfer and the actual training loader.

The samples/s view gives a different perspective on crop size. H100 CUDA's best
random medians are about **16,289, 10,127, and 7,804 samples/s** at 128², 192², and
256²; its best baked medians are **69,914, 83,309, and 71,263 samples/s**. More image
bytes per second can accompany fewer completed samples. The
[crop protocol](#random-crops-and-baked-samples) also explains the fixed CPU
input-buffer limit and the extra read/decode groups at 128².

Lower IOPS can accompany faster sample delivery. At 128² on H100 CUDA, the fastest
random setting uses about **258 kB of storage reads and 7.68 storage operations per
sample**. The fastest baked setting uses about **36 kB and 0.275 operations**.
The full report retains these storage counters, along with image throughput and
compression, for every setting.

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

## Earlier paired CPU and H100 cold-read results

The [November 6 paired study](pareto/index.html#study=81fe76d51bbbd3212ed6)
contains **252 settings and 616 executions**: nine layouts, both read patterns,
the complete seven-asset mixture, six dataset omissions, and both CPU and CUDA.
Each setting has two process repetitions. Both backends ran on one H100 80 GB
node with an Intel Xeon Platinum 8462Y+, 32 allowed logical CPUs, 16 CPU decode
workers, and 16 file workers. Its storage system is XFS on an array of eight
7.68 TB Intel SSDPF2KX076T1O PCIe NVMe drives. The run reused the verified
769.17 GiB preparation and completed in 79 minutes 33 seconds, including local
staging and all 42 native tests.

These are all-source image-throughput medians in **GiB/s**, comparing the same
layouts on that machine. The table includes every layout on at least one observed
frontier. Blosc block sizes in the labels are requests; actual block sizes from
the encoded headers are retained in the report.

| Layout | Compression ratio | CPU crops | CUDA crops | CPU whole frames | CUDA whole frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| Blosc-LZ4, 256 KiB chunks / 256 KiB blocks | 1.433× | 1.025 | 1.190 | 4.245 | 4.805 |
| Zstd, 1 MiB chunks | 1.441× | 0.922 | 0.812 | 4.108 | 3.570 |
| Blosc-Zstd, 256 KiB chunks / 16 KiB blocks | 1.475× | 0.944 | 1.187 | 4.002 | 4.959 |
| Blosc-Zstd, 256 KiB chunks / 256 KiB blocks | 1.504× | 0.961 | 1.157 | 4.022 | 4.773 |

On CUDA, the two Blosc-Zstd layouts trade whole-frame throughput against
compression. Crop medians for Blosc-LZ4/256 KiB and Blosc-Zstd/16 KiB blocks are
within 0.3%, with overlapping repeat ranges. CUDA does not improve every layout:
Zstd/1 MiB is slower than its paired CPU measurement in both read patterns.
The CPU values here come from the same allocation as the CUDA values; the
November 5 EPYC study below describes a different machine.

Across all settings, the median min–max throughput spread was 1.12% of the
median, with a maximum of 12.08%. Bracketing raw-control spreads ranged from
0.70% to 7.87%. Settings have two repetitions; control ranges span four
bracketing executions. These are observed min–max ranges, not confidence
intervals or a reliable ordering of close settings.

All **1,374 measured passes** verified zero resident shard-data pages and
recorded storage reads. Peak overlapping file read calls reached 16 in every
execution. IOPS ranged from 29,945 to 134,864 across executions. Adjacent idle
traffic reached 180 IOPS, at most 0.46% of the corresponding measured rate.
For CUDA whole-frame Blosc-Zstd with 256 KiB chunks and 16 KiB requested blocks,
the medians were **4.959 GiB/s image throughput**, **3.500 GiB/s storage bandwidth**,
and **48,554 IOPS**, with **1.475× compression**.

The report includes each dataset omission and its own uncompressed control.
The [retained JSON](pareto/data/81fe76d51bbbd3212ed6.json) contains every observation,
cache check, device counter, source geometry, and source, query, and binary hash.
This completes the planned nine-layout paired study; the 22-layout matrix has
not been measured.

## Earlier mixed-source CPU results

The [November 5 CPU study](pareto/index.html#study=542501404495a92083a9)
contains **126 settings and 308 executions**, using all seven microscopy assets,
nine layouts, two read patterns, and six dataset omissions. Each setting has two
process repetitions. It ran on an AMD EPYC 9655P with 32 allowed logical CPUs,
16 decode workers, 16 file workers, and eight 7.68 TB Intel PCIe NVMe drives.
The prepared files total 769.17 GiB, representing 1008 GiB of uncompressed pixels.

For the complete source mixture, two layouts form the observed frontier in both
read patterns:

| Layout | Compression ratio | Crop image throughput (GiB/s) | Whole-frame image throughput (GiB/s) |
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
IOPS ranged from about 32,000 to 141,000 across executions;
these rates describe the measured requests and include device activity outside
the process. The adjacent idle rate reached 455 read IOPS. For all-source
whole-frame Zstd/1 MiB reads, the medians were 70,375 IOPS, 7.343 GiB/s storage bandwidth, and
5.579 GiB/s image throughput.

Omitting DynaCell changes the highest-median layout to Blosc-Zstd/256 KiB for
both read patterns. Omitting BBBC010 changes whole-frame Zstd/1 MiB throughput
by only −0.47%, while its compression falls 10.14%, from 1.441× to 1.295×.
The sensitivity panel shows each mixture's compression ratio and speed vs its
own uncompressed control, including the all-source mixture as a reference.
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
comparison of cache state. The paired cold H100 study above uses the larger,
mixed-source protocol; the full 22-layout matrix remains unmeasured.

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

| Blosc block size | Image throughput (GiB/s), median | Observed range | Compression ratio |
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

## Random crops and baked samples

The current [`bench/studies/microscopy.json`](https://github.com/nclack/damacy/blob/bench/microscopy-pareto/bench/studies/microscopy.json)
defines the smaller study whose [measured results are above](#crop-results).
It compares 128 × 128, 192 × 192, and 256 × 256 crops from the complete seven-input
mixture. Whole-frame reads remain in the historical reports.

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

Each crop size and access pattern has its own Pareto frontier. The **Crop size and
samples/s** view compares completed crops per second at the fixed byte budgets.
It converts the image throughput using the exact source composition of each query
stream. A line follows one encoding recipe, not a frontier across different tasks.
Compression ratio still describes the actual stored arrays: random arrays have
equal logical bytes per source; baked arrays have equal sample counts. Their
compression ratios therefore weight native source types differently. The paired
read rates use the same source samples and byte counts.

The cold-pass protocol below applies unchanged. Every timed pass begins with zero
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

## Mixed-source microscopy study

The retained [`bench/studies/microscopy-mixture-v2.json`](https://github.com/nclack/damacy/blob/bench/microscopy-pareto/bench/studies/microscopy-mixture-v2.json)
uses every asset in the version-3 [corpus manifest](https://github.com/nclack/chucky-benchmarks-data).
The current seven inputs are OpenCell DNA and protein, BBBC010 brightfield,
JUMP-Scope fluorescence, DynaCell phase, COSEM EM, and BBBC022 MitoTracker.
The November 5 CPU and November 6 paired H100 studies above use this geometry
and protocol.
The older single-input pilots retain their original, smaller inputs.

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

That earlier matrix compares raw bytes, Zstd, Blosc-Zstd, and Blosc-LZ4; compression
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
used when computing image throughput. Compression aggregates the equal 16 GiB
source volumes by summing logical and stored bytes; read weights are equal
sample counts, which give unequal source-byte contributions across types.

Leave-one-out cases remove a whole dataset collection and retain all remaining
coordinates and their relative ordering. Both OpenCell channels are removed
together. The pass contains less work after an omission, with the same work per
retained input. The report compares compression ratio and speed vs each mixture's
uncompressed control (256 KiB chunks). Speed is image throughput divided by control
image throughput, so the mixture's source-byte weighting cancels in that ratio.
Changes also reflect the reduced number of source files; they are sensitivity measurements, not causal
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
retained in `bench/studies/microscopy-pilot-v1.json` and
`bench/studies/microscopy-mixture-v2.json` for reproducing old data.


## Reproduce a study

Use a materialized version-3 corpus with `manifest.json` and the declared raw
files. Planning reads metadata only. The default definition is the new crop study; use
`microscopy-mixture-v2.json --pilot` to reproduce the completed mixed study:

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
The workflow summary links to the preview. Production docs and benchmark histories
keep their existing URLs. Every docs build also uploads a `docs-site` artifact;
download and unzip it, then run `python -m http.server` from that directory.
Fork pull requests provide the artifact without publishing to Pages.
