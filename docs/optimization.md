# Optimizing damacy: what we learned

This is a story in two arcs. The first is optimizing damacy on a dataset chosen to
be a clean target: what the pipeline does, the choices that mattered, and the
measurements behind them. The second is what happened when we went looking for the
*next* optimization across the whole training corpus — the bottleneck moved out of
the loader and into the data. We ruled out the remaining code-side fixes (a chunk
cache chief among them), and that was the signal to stop tuning the loader and turn
to how the data is stored.

All numbers are from real benchmark runs on an L40 GPU node reading fast networked
storage (16 parallel connections), unless noted. (For this work the GPU tier doesn't
matter: the runs are limited by reading and decompressing, not by GPU math, so an
L40 reproduces the same numbers a bigger GPU would.)

## 1. What damacy does, stage by stage

damacy feeds a training loop with small crops ("patches") sampled at random from a
large collection of compressed image arrays on disk. The arrays are stored in the
zarr v3 format: each array is split into **chunks** (the smallest block stored and
compressed as one unit), and many chunks are often packed into one file called a
**shard**. To serve one patch, damacy figures out which chunks it overlaps, reads
those chunks off storage, decompresses them on the GPU, and copies the wanted crop
into the output batch.

The benchmark meters the pipeline as a sequence of stages. The two that ever
dominate are **reading** and **decompressing**; everything else is cheap.

| stage | what it does | runs on | limited by |
|---|---|---|---|
| plan | works out which chunks each patch needs; merges and orders reads | CPU | nothing (microseconds) |
| **io** | reads compressed chunk bytes off storage | CPU worker pool | storage throughput + latency under load |
| input_transfer | copies compressed bytes to the GPU | GPU copy | nothing (~25 GB/s, never the wall) |
| **decode** | decompresses chunks on the GPU | GPU | volume of data it has to decompress |
| assemble | copies the wanted crop out of each chunk into the batch | GPU | nothing (~hundreds GB/s) |

How the two heavy stages work:

- **Reading** uses a pool of worker threads, each doing one blocking read at a
  time. The number of threads (`n_io_threads`) sets how many reads are in flight at
  once. Adjacent chunk reads within a shard are merged into one larger read, up to
  a cap (`max_read_op_kb`). Looking up file/array layout uses a separate, faster
  async path, sized by `metadata_io_concurrency`; small caches keep that layout
  info so it's a one-time startup cost.
- **Decompressing** happens on the GPU. The planner reads ahead by
  `lookahead_samples` patches to find chunks to prefetch, and keeps the in-flight
  working set inside a memory budget (`max_gpu_memory_mb`), processing in waves.

Two numbers describe how efficient a run is, and the rest of this report leans on
them:

- **read waste** = bytes read off storage ÷ bytes the patches actually need.
- **decode waste** = bytes decompressed ÷ bytes the patches actually need.

A value of 1 is perfect. 100 means 99% of the work is thrown away. These are high
whenever a chunk is as big as the crop or bigger, because you must read and
decompress the *whole* chunk to get any pixel inside it.

## 2. Starting point: dynacell, picked because it was already well chunked

dynacell was chosen as the first target on purpose. It was known to be well chunked
(1 MB chunks packed into shards, decode waste only ~4.3×), so optimizing it would
isolate the **reading** path with nothing on the decode side to muddy the picture —
a clean target for the storage work. That held: on dynacell the GPU sits idle most
of the time and the wall is entirely the reading stage (decode busy only 6–27% of
the run; reading is the bottleneck).

The root cause was the storage path, not damacy. The original location was a
network filesystem reached over a single connection; many small concurrent reads
queued on that one connection and latency ballooned. Moving to a mirror reached
over 16 parallel connections, and then reading in a way that uses that parallelism,
was most of the win.

We could not properly evaluate the GPU-direct read path (cuFile/GDS): it needs
storage reached over RDMA, and we have no such mount available. On the network
filesystem we do have, cuFile silently falls back to a copy-through-host
*compatibility* mode, which is poor (~0.10 GB/s). So the only firm conclusion is
that compatibility mode isn't useful here — true GDS is still unevaluated, and is
worth revisiting if an RDMA mount becomes available.

## 3. The levers that mattered, and the measurements

Stacking the changes on dynacell, lowest to highest throughput. Each row adds one
lever to the row above.

| step | change | GB/s | vs start |
|---|---|---|---|
| start | single-connection storage, defaults (16 read threads, no merging) | 0.38 | 1.0× |
| + fast storage | mirror with 16 parallel connections | 0.58 | 1.5× |
| + merge reads | merge adjacent reads up to 4 MB | 1.10 | 2.9× |
| + more read threads | 16 → 64 in-flight reads | 2.03 | 5.3× |
| + code improvements | interleave reads across shards, deeper read-ahead, open files in the read workers (#146, #147, #149) | 2.82 | 7.4× |
| + host buffering | buffer more decoded waves in host memory (`host_buffer_waves=4`) | 4.85 | **12.8×** |

The last row (host buffering) is the newest lever and is still being characterized;
4.85 GB/s is the best observed, not yet a settled number. Everything above it is
solid and repeatable.

The two read levers came from controlled sweeps (one knob at a time, same data):

**Merging reads** — bigger merges cut the number of separate reads in half and
saturate at 2 MB (dynacell's runs of adjacent chunks top out there, so 2 MB and up
all execute the identical read plan; the spread above 2 MB is run-to-run network
noise):

| merge cap | reads issued | GB/s |
|---|---|---|
| 512 KB (default) | 55,399 | 1.40 |
| 1 MB | 50,295 | 1.54 |
| 2 MB | 27,857 | 2.10 |
| 4 MB | 27,850 | 1.91 |
| 8–32 MB | 27,848 | 1.83–2.01 |

**Read threads (in-flight reads)** — scales nearly linearly to ~16, with the knee
around 48–64. The default of 16 left ~1.8× on the table:

| read threads | 1 | 4 | 8 | 16 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|
| GB/s | 0.14 | 0.47 | 0.72 | 1.10 | 1.57 | 1.97 | 2.03 |

Two knobs turned out **not** to matter and are "set once and forget": metadata
concurrency (flat from 32 to 256, slightly worse at 256) and the chunk-size cap
(flat from 1 to 4 MB — it only needs to be large enough to accept the data's
chunks). The confirmed settings: 64 read threads, 64 metadata concurrency, 4 MB
merge cap, 2 MB chunk cap.

## 4. Looking for the next win: across the corpus, most datasets are limited by *decompressing*, not reading

With dynacell fast, the question was where the next optimization was, so we pointed
the tuned pipeline at the rest of the training datasets. The bottleneck flipped. On
most of them the decode stage is busy ~99–100% of the run and the read stage is
nearly idle — the opposite of dynacell. The storage tuning above does almost
nothing for them, because storage is no longer the wall.

The same run, two regimes side by side:

| | dynacell (tuned) | allencell-static |
|---|---|---|
| throughput | 4.85 GB/s | 0.39 GB/s |
| read stage busy | ~100% (the wall) | 40% |
| decode stage busy | 27% | **99% (the wall)** |
| decode waste | 4.3× | **116×** |

The puzzle "decode runs at 51 GB/s but throughput is 0.4 GB/s" resolves cleanly:
throughput ≈ decode rate ÷ decode waste = 51 ÷ 116 ≈ 0.4. The GPU decompressor is
fast; it's just decompressing ~100× more than the patches need, because each tiny
crop forces decompressing a whole oversized chunk.

This is why it's a data-formatting problem, not a pipeline-tuning problem. We also
confirmed that decode waste is predictable from the file layout alone (chunk shape
vs array shape vs patch shape), with no benchmark needed — the prediction matched
the measured values within ~10%, which let us survey the whole corpus cheaply.

## 5. The dataset landscape

Reading the file layout for every dataset the current training config uses (32
datasets), plus the raw 3D versions of several. Most of the corpus is already in
good shape; the problems concentrate in a handful of datasets and fall into a few
clear categories.

| category | count | what's wrong | example (decode waste) |
|---|---|---|---|
| well-formed | 23 | small sharded chunks, well under 1 MB | rxrx, lincs, periscope, opencell (1–5×) |
| over-declared / mostly empty | 4 | chunk has extent > 1 in dimensions the array barely fills; most of the chunk is padding | allencell-static (128×), allencell-nuclear (131×) |
| coarse, unsharded | 6 | 32–64 MB chunks, one file each — as big as or bigger than the crop | dynacell-MIP (33×), jacobo-MIP (31×) |
| coarse 3D | 6 | big 3D chunks read by a small 2D crop | raw allencell / liberali / emt (76–163×) |

The 23 well-formed datasets are the template: sharded files holding small chunks,
well under 1 MB. Two things to keep straight about what "well chunked" means here:

- **The "dynacell" the config actually trains is the flattened (MIP) version**, with
  64 MB unsharded chunks and decode waste of 33× — *not* the well-chunked raw
  dynacell we tuned in §2–3 (1 MB sharded, 4.3×). They are different data with
  different layouts. The good behavior we measured belongs to the raw, sharded form.
- **Small in the right way, not just small.** Two rules. First, the chunk should be
  *smaller than the patch* — comfortably under 1 MB, and as small as ~64 kB before
  compression starts to suffer — so a patch is built from many fully-used chunks plus
  thin borders, instead of being forced to pull in a whole chunk it barely uses.
  Second, only give a dimension a chunk extent greater than 1 where neighboring
  values along it actually correlate, so compressing them together pays for itself:
  x and y always, time sometimes, depth sometimes, channels rarely. An extent the
  patch reads only a slice of is wasted on every read — it's why the well-sharded raw
  dynacell still shows 15.9× at a single-plane, 16-frame crop (its 1 MB chunk spans 4
  depth planes the crop ignores and is as wide as the crop, so unaligned crops
  straddle it). Smaller chunks, with extent only where it compresses, drive that
  toward 1×.

The surveyed problem datasets, measured:

| dataset | crop | throughput GB/s | patches/s | decode waste |
|---|---|---|---|---|
| chammi75 | 2D still | 0.25 | 1056 | 5.4× |
| allencell-static | 2D still | 0.39 | 1669 | 116× |
| jacobo-MIP | 16-frame | 0.21 | 56 | 28× |
| allencell-dynamic | 16-frame | 0.14 | 38 | 20× |

## 6. Knowing when to stop: the lever is the data, not the loader

With dynacell tuned and the corpus surveyed, the real question was whether there was
more to win inside damacy, or whether the loader was done for this workload. We
weighed the two obvious code-side fixes, and both were dead ends — which is what
told us to stop and turn to the data.

**Could a cache help?** Decoding is fast: it produces ~51 GB/s of output. The waste
isn't slow decoding — it's that the same oversized chunk is decompressed again for
every crop that lands in it. So the obvious idea is to decode each chunk once and
reuse it. It doesn't work for training. Crops are sampled in shuffled order, so a
given chunk isn't touched again until a whole pass through the data later — long
after any sane cache has evicted it. To actually capture reuse, the cache would have
to hold essentially the entire dataset *in decompressed form* — terabytes for these
sets (allencell-static alone is ~3 TB decompressed). That isn't a cache; it's keeping
the whole corpus resident. Spilling it to host memory, or letting it page
automatically between GPU and host, doesn't rescue it: with no reuse to exploit you
just stream the full dataset across the link to the GPU on every pass. And it fights
a hard requirement — while training, damacy's GPU memory has to stay small and
predictable so it doesn't compete with the model. The redundant decoding can't be
cached away; it can only be designed away, in how the data is stored.

**Could a different reader help?** We built a matched CPU benchmark (tensorstore, the
leading CPU reader) reading the exact same crops, to weigh the GPU-vs-CPU choice on
the current, un-reformatted data. Comparing on patches/second (the apples-to-apples
number; the two count bytes differently):

| dataset | damacy GPU | tensorstore CPU (best threads) | winner |
|---|---|---|---|
| chammi75 | 1056 | 2604 (32) | CPU 2.5× |
| allencell-static | 1669 | 325 (16) | GPU 5.1× |
| jacobo-MIP | 56 | 59 (32) | tie |
| allencell-dynamic | 38 | 203 (32) | CPU 5.3× |

No clear winner — it tracks the data, not the backend. The GPU wins big where the
wasted volume is mostly padding it can blow through cheaply (allencell-static); the
CPU's 32 threads win on small real chunks (chammi) and where there are too few
arrays for damacy to parallelize (allencell-dynamic, only 2 arrays — a damacy
weakness now tracked as a bug). (Caveats: this under-counts damacy for training,
since its output lands on the GPU and frees the CPU while tensorstore reads to host
and still owes a copy; it's all on the badly-formatted data; and small-array
datasets are noisy.) Once the data is reformatted so decoding stops being the wall,
the contest moves to reading and getting data onto the GPU — damacy's strengths.

Both dead ends point the same way. For the decode-bound datasets the bottleneck is
the data's on-disk shape, and no loader engineering — caching, a faster decoder, a
different backend — moves it. That was the signal: the code work was essentially
done for this workload, and the productive thing was to fix the data.

## 7. What it adds up to

The work drew a clean line between two problems:

1. **Reading-limited data** (well-chunked, like raw dynacell): a loader problem, and
   solved — the storage and pipeline tuning in §3, ~13× from 0.38 to 4.85 GB/s, with
   a settled config.
2. **Decompress-limited data** (most of the corpus): *not* a loader problem. No
   pipeline knob, cache, or backend swap fixes it; the fix is reformatting the data
   into small chunks (well under 1 MB), with extent only in the dimensions that
   compress well together. For this workload the loader is essentially done — the
   next gains come from the data.

The scope of that data work is small and knowable: only ~9 of 32 datasets need
reformatting, the well-formed 23 already show the target shape, and we can predict
the improvement from layout before moving a single byte.

Open threads: a reformatting spec (target chunk/shard shapes per data type, with
predicted "after" numbers); damacy's low-array-count parallelism limit (issue #154);
and finishing the characterization of the host-buffering lever from §3.

---
*Numbers come from benchmark runs produced by the harness in `bench/` (the scenario
runner `bench/run.py` and the per-stage `bench/report.py`), with synthetic and real
scenarios. The corpus survey, the CPU comparison (`bench/tensorstore_bench.py`), and
the layout scan were run with cluster-side tooling against the training datasets;
the methods are reproducible from the bench harness.*
