# Data reformatting plan (draft)

The corpus survey (see the optimization report in `docs/optimization.md`) found that
most training datasets are decode-bound: a small crop forces decompressing a whole
oversized chunk, so decode waste runs 20–130×. No loader change fixes this — the fix
is re-chunking the data. This plan gives a target chunk/shard shape per data
archetype and the **predicted** decode waste after reformatting, so the win is known
before any bytes move. Predictions come from a what-if calculator that counts the
chunks a random crop touches against the real array and patch shapes; it reproduces
the measured "before" values, so the "after" values are on the same footing.

"Decode waste" = bytes decompressed ÷ bytes the crop actually needs (1.0 = perfect).

## The principle

Chunks should be **small, and small in the right way** — not "matched to the read":

1. **Smaller than the crop**, in the 64 KiB – 1 MiB band. Below ~64 KiB compression
   starts to suffer; the problem now is the opposite — chunks as big as or bigger than
   the crop, so a crop can't avoid pulling in a whole chunk it barely uses.
2. **Extent > 1 in a dimension only where neighboring values correlate** enough that
   compressing them together pays off: **x and y always**, **z in true 3D**, **time
   sometimes**, **channels rarely (split them — chunk per channel)**. An extent the
   crop reads only a slice of is wasted on every read.

A residual 2–4× remains even at the ideal chunk — a random crop straddles chunk
borders, and source dtypes narrower than the f32 output shift the ratio. That floor
is fine; it's the well-formed band. The goal is to remove the 20–130× excess, not to
chase 1.0×.

## Targets per archetype

| archetype | training crop (TCZYX) | target inner chunk | typical size | why |
|---|---|---|---|---|
| static 2D | `[1,1,1,256,256]` | `[1,1,1,256,256]` | 64–256 KiB | x/y only; t=1 (still), c=1 (channels independent) |
| video / temporal | `[16,1,1,256,256]` | `[1,1,1,256,256]` | 128–256 KiB | x/y only, c=1, t=1 by default — a 16-frame crop reads 16 small fully-used chunks |
| 3D z-range | `[1,1,16,256,256]`-ish | `[1,1,4,256,256]` | ~0.5–1 MiB | x/y + a small z extent (3D has z coherence), fully used by a z-range crop |

- **Time extent is a compression-vs-read tradeoff.** `t=1` minimizes read waste; a
  16-frame crop reads 16 small fully-used chunks. A `t=4` chunk only pays off if
  consecutive frames compress enough better together to beat the small extra read
  waste it adds (jacobo: 4.0× → 4.7×). Default to `t=1`; raise it only with a measured
  compression gain.
- **Channels split** (`c=1`): channels in these sets don't correlate, so stacking
  them only enlarges the chunk for nothing.
- **Spatial = the crop (256)**, not larger: 512 roughly doubles the waste
  (allencell-static 2.0× → 3.6×); below ~256 on a narrow dtype drops under 64 KiB.

## Predicted win, per dataset

decode waste before (measured) → after (computed at the target chunk). These nine are
the datasets above the well-formed band (≲5×); the other 23 already meet the target.

| dataset | archetype | before | after | target chunk | size |
|---|---|---|---|---|---|
| allencell-nuclear | static 2D | 131× | **1.9×** | `[1,1,1,256,256]` u16 | 128 KiB |
| allencell-static | static 2D | 128× | **2.0×** | `[1,1,1,256,256]` u16 | 128 KiB |
| dynacell-MIP | video | 33× | **4.0×** | `[1,1,1,256,256]` f32 | 256 KiB |
| jacobo-MIP | video | 31× | **4.0×** | `[1,1,1,256,256]` f32 | 256 KiB |
| allencell-nucmorph | video | 22× | **2.0×** | `[1,1,1,256,256]` u16 | 128 KiB |
| emt-timelapse | video | 22× | **2.0×** | `[1,1,1,256,256]` u16 | 128 KiB |
| cellstate-phenotyping | static 2D | 9.0× | **4.0×** | `[1,1,1,256,256]` f32 | 256 KiB |
| chammi75 | static 2D | 6.2× | **1.0×** | `[1,1,1,256,256]` u8 | 64 KiB |
| liberali-ls1 | video | 6.1× | **2.0×** | `[1,1,1,256,256]` u16 | 128 KiB |

The big wins are the two over-declared allencell sets (128–131× → ~2×) and the four
video MIPs (22–33× → 2–4×). cellstate / chammi / liberali-ls1 are minor (already 6–9×;
worth doing only when their stores are touched anyway). liberali-ls1 is a good
illustration of rule 2: its current chunk is `[84,1,1,69,147]` — a huge time extent
the 16-frame crop barely uses, plus spatial smaller than the crop.

## The raw 3D variants — read them with a 3D crop first

The raw (non-MIP) 3D stores show 76–163× in the survey, but that number is misleading:
it's the *2D* crop applied to 3D data — reading one z-plane out of a chunk that spans
the whole z stack. Training is moving to raw 3D with z-range crops, and **read with a
z-range crop the current raw chunks are only ~5–10×**, and the target `[1,1,4,256,256]`
brings them to ~2–5×:

| dataset | crop | current chunk → waste | target `[1,1,4,256,256]` |
|---|---|---|---|
| jacobo raw | `[1,1,16,256,256]` | `[1,1,73,128,128]` → 10.2× | **4.7×** |
| allencell-static raw | `[1,1,16,256,256]` | `[1,1,66,128,128]` → 4.6× | **2.4×** |
| raw dynacell (template) | `[1,1,16,256,256]` | `[1,1,4,256,256]` → 4.7× | 4.7× (already good) |

So for the raw-3D move, two things change together: use a z-range crop (most of the
apparent problem), and chunk with a small z extent like raw dynacell already does
(the rest). Raw dynacell is the worked example to copy.

## Sharding: ≥1 GB, and think in file count, not a byte ceiling

Small inner chunks must be packed into **shards** (one file holding many chunks), or
you trade decode waste for a file-count explosion. **Use shards of ~1 GB uncompressed
or larger** — thousands of inner chunks per file — to amortize per-file open and
metadata overhead and keep reads sequential. (dynacell's shards are ~1–2 GB
uncompressed, dozens per array, packing `[1,1,4,256,256]` inner chunks.)

People ask what the *maximum* shard size should be. The useful constraint isn't a byte
ceiling — it's a file-count floor. A shard is the unit of parallel-read independence:
damacy issues many chunk reads at once and interleaves them across shards, so reads to
different shard files run in parallel while reads inside one shard tend to serialize.
What matters is that a batch's in-flight reads span **enough independent files to fill
the read pool** (dozens of concurrent reads). In practice:

- **Large arrays** (multi-GB timelapses, 3D, whole-slide): 1 GB shards already yield
  dozens of shards per array, so reads spread across files on their own — no upper
  limit needed. Just don't put an entire large array in a single shard.
- **Small arrays** (< ~1 GB — most 2D FOV sets): the array is one sub-1 GB shard, which
  is fine; parallelism comes from sampling across the dataset's many arrays.
- **Danger zone:** few arrays *and* few shards together (e.g. a 2-array dataset) can't
  fill the read pool no matter the chunking — the low-array-count limit measured in
  issue #154, a property of the dataset rather than the shard choice.

So: **≥1 GB uncompressed per shard, with enough shards-and-arrays in the in-flight
window to cover your read concurrency.** That reframes the confusing "maximum shard
size" question as a file-count floor, which is the thing that actually drives
throughput.

## Caveats / verify before committing

- **Check the compression ratio after re-chunking**, per dataset — the 64 KiB floor is
  a guideline, not a guarantee; some data compresses worse at small chunks.
- **The after-numbers are predicted** from geometry. Confirm with one real bench run
  (`bench/run.py`) on a single reformatted dataset before doing the whole corpus.
- **This is an offline data migration** — rewrite each store once — not a loader change.
- **Priorities by win size:** allencell-static / allencell-nuclear first (128–131× →
  ~2×, and they're large many-array datasets), then the video MIPs (20–33× → 2–4×).

## Method

The before/after numbers are computed (no GPU, no data reads) by sampling random crop
placements against each dataset's real array and chunk shapes and counting touched
chunks — the same definition the bench reports. The corpus-layout scan and the
chunk what-if calculator live with the cluster-side survey tooling; either can produce
the "after" number for any candidate chunk shape, so alternatives can be compared
before moving data.
