# dev log

## 2026-06-01

Cleanup of prefetcher seems like it's converging.

- [x] io_uring for prefetcher
- [ ] api: dependency injection - stores, prefetcher

## 2026-05-18

Investigating some of the threading primitives that showed up in a perf scan.
Maybe a futex etc will help.

Found some other good leads in store_fs.


## 2026-05-17

Getting things back into consistently using the platform layer. Got a speed
up out of this which is weird.

## 2026-05-16

Looking to measure backpressure, so added a hold time to the benchmark.
Should simulate training holding on to the batch. Some results:

```
hold(ms) block(total ms, 50 batches)
200      403
300      355
400      350
500      329
````

I'd really like that block time to go to zero.

Also looking into chunk coalescing. Folks reported some high variance in nfs
latencies, so this might help. It would also help with resilience in the face
of small chunk sizes. May think about coalescing at page granularity.

- [x] bench with a store with some noisy latency

decomp masks the upstream pipeline so it's hard to get a read on performance
at some point. Added a flag to treat chunks as "fill chunks" post io. These
use the fill value to populate the batch, so I don't need to decomrpess but
the pipeline still runs.

## 2026-05-15

I've been using a more github based workflow, so not using this doc as much.

Working on GDS and gpu-based blosc parsing.

GDS _requires_ gpu-based blosc parsing. So I'm doing them together.

Re blosc header parsing, I realized (a) all the chunks headers are the same
for the most part and (b) with blosc-zstd there are some some assumptions we
can make - I think it's basically just 1 substream.

- [x] sanity check the parsing kernel

Ritvik saw some large variance in latencies over NFS. Should take a look at
being robust to that.

- [x] chunk coalescing.

Found a better design for the parsing kernel. Adding that now.

- [x] remove the cpu parsing once it looks like gpu is better.

decompress and assemble are not overlapping
      
## 2026-05-08

Making the python interface nicer. Adding typing, linting and formatting w
ruff + pyright.

- [x] tests/coverage
- [x] autodoc
- [ ] wheel
- [x] readme examples
- [x] !double check context binding behavior.
- [x] readme should show some non-trivial crops. mention all crops have to
      be same size
- [x] does the api need to be able to take a (cuda0 stream as input
- [x] !FIX store root in the python api
- [x] !FIX python users expect all pushed input to get consumed

I forgot about nvidia's DALI. I need to evaluate positioning wrt that.
Added a TODO about comparative benchmarking. Looks like there have been some
hackathon projects wrt zarr on DALI. Should look at that.

- [x] (WONT DO) think about caller supplied memory. Batch memory reuse. The
      pointer might pin the device so there's no ambiguity.
- [x] maybe move different ddp examples to docs and off the readme

Ran `try-nvcomp` on this machine, and I get 42 GB/s, so we've got lots of
headroom on this machine.

- [x] get blosc chunk header parsing on to the cpu

cleanup

- [x] refactor long files

## 2026-05-07

continuing on blosc

- [x] performance pass on blosc kernels
- [x] benchmark scenarios - all blosc + mixed
- [x] evaluated eliminating the d2h after parsing the blosc headers
- [x] test mixed codec sources

looks pretty slow - or is it?

- [x] review all the limits - aggregate to a private header.

blosc chunk headers are parsed on gpu but that is probably a mistake. Just
parse on cpu w a threadpool avoids the d2h, error handling better.

Testing on the cluster found (a) need to handle device contexts better
and (b) need to get proper dtype support in. So I added that.

## 2026-05-06

4.8 GB/s on the cluster. Should be faster.

They tried the python side but ran into the lack of blosc support.

Adding that on the cpu side is depressing; waves could be mixed.

Looking into blosc1 on gpu. might not be that bad.

There's an awkward sync that needs to happen for blosc so that nvdecomp gets
the right input. Later I might need to take a look at the kernel primitives
for nvcomp.

## 2026-05-05

Goal for today would be to actually try this out on the cluster. Will also
start on the python interface. It would be nice to get this into a state
where I can hand it off.

- [x] zarr_metadata.c - separate parsing and validation

## 2026-05-04

Question: which way to do the step optimization? Maybe from the end to the
beginning so we now what the downstream bottlenecks are as we go.

Focus on making the benchmarking nice first though. Reviewing for accuracy
and then: 

- [x] refine bench scenario(s)
- [x] need to do a better multi-zarr setup for bench/test
- [x] benchmark output as json

Working on specing scenarios as json, making benchmark out json, and refining
the base scenario. I'd like 256kB chunks before compression. 3d zarrs with at
least 4 shards each, maybe 512x512x512 each. sample random in bounds aabb. For
building the batches, I want each "Sample" to be 64x64x64. Batch of about 1 GB.

- [x] deal with the "max chunks per batch" - wrong parameterization
- [x] review/reduce comments

Need to speed up a`assemble.cu`.

- [x] assert that chunk sizes are the same for all lod's from the same zarr.
      This will be a design constraint that can probably be relaxed in practice.
- [x] Looks like the code is clipping chunks on load, which isn't right. fix

Much better. Got assemble from 1 GB/s to 80 GB/s.
      

## 2026-05-03

Working on design/interface some more.

One interesting note: I was thinking about multiplexing to gpus but that doesn't
work for ddp since that's normally one gpu per process (just easier to do in
python that way). Could think about a loading process, but then how do you
switch to cufile etc. Also, centralizing the caches just saves relatively cheap
memory though it might save some loads. Anyway, looks like it would have to look
more like a peer-exchange if you wanted to save io. Almost certainly not
worth it (especially if you have a network layer cache).

Got a basic end-to-end pipeline up. Still needs a lot of work.

- containerof pattern for lru elements?
- review the kernel 

## 2026-05-02

Reviewing the generated code.

Using ngff-zarr to generate data. Might switch to chucky at some point if
speed/memory become a thing. I also want to dogfood a little. Could probably
work on integrating via flake which would be fun.

Revised the json interface to (a) use a query structure and (b) parse the
string from scratch, more or less, to satisfy a query. This way it's
zero-copy zero-alloc. If it's slow, then later we could probably cache by
query prefix. Added tests, fuzzing and coverage.

Setting up the file io so that it's mmap for metadata and direct (unbuffered)
io for reads. This means that we'll need a layer that coalesces chunks by
page and then produces the right chunk pointers later.

Mappings:
- `uri` -> `array_id`
- `array_id` -> `layout`: Layout is `(shape,shards,chunks)`

Steps, given a collection of `(uri,bbox)` requests
1. Read metadata from collection of uri's. What do we need to pull out?
2. Map bbox's to `(shard id, chunk id)`
3. ...

Will want to stream read bytes up to the gpu. What is the indexing structure?

The stages are still not clear to me though it seems they should be.

## 2026-05-01

Spun up a basic example.

## 2026-04-29

Simplify, try to minimize complexity introduced from transforms.

Samples are axis-aligned to their sources. Maybe scaled. Api would look like
a slice. Maybe...actually, samples all have a uniform bbox. Source may have
a different bbox. Then the task would be how to resample from source to dest.

API looks like generating `(uri,aabb)`

I think the key thing here is streaming the samples for batching so that we
can start requesting chunks for the next batch while one is rendering. That
sample stream could get transformed to a coalesced chunk-request stream.
As chunks get read in, they need to get indexed somehow for retrieval during
rendering and we may need some kind of LRU.

Did some benchmarking of kvikio. The c++ library uses a threadpool at read
and has no trouble getting full nvme bandwidth for sequential read. The api
is a pretty bare parallel pread. No real format support on the c++ side.

Should probably bench the python side. It looks like that's where all the
interesting stuff happens. nvcomp is done through numcodecs.

Ok. kvikio is interesting in that it has pretty strong integration with the
python ecosystem afaict. It does pretty good on bulk array reads but there's
a lot of read amplification when reading from subregions.

## 2026-04-20

Transform steps

1. `r_out -> source_id` SELECT
- requires transforms
- in general, the sample could map to many sources. We may be able to assume
  one source by contract here if it helps.
- this is a point-into-convex polygons (non aabb rects) or something. Anyway
  this part needs thought.

2. `r_out -- T(source_id) --> r_source` MAP
- source id is used to look up the transform
- there's some transform composition here

3. If we did have multiple sources, the last step would be a REDUCE

Source selection is expensive. We're probably actually talking about selecting
from the set of chunks...from multiple sources.  

## 2026-04-16

API design

Batches are homogeneous across dimensions: if the samples are (t,c,z,y,x)
all samples have the same shape in those dimensions.

If we need to load s samples, the output array is (s,t,c...). That is, the
samples are just concatenated on the outer dimension.

The range is always the same for each sample. We need to specify the source
for the sample and a transform mapping the range back into the source domain
- the index space of the source array.

It will be important to stream batches in.

## 2026-04-11

[DLPack](https://github.com/dmlc/dlpack) is probably good to target as an array
interchange between c and jax/pytorch etc.

## 2026-03-31

Loading tensors from zarr stores on the gpu is much slower than it should be.

As far as I can tell, nvcomp can decompress chunks at around 100 GB/s on
H100's. But kvikio, the library developed by RAPIDS reportedly only goes about
2GB/s.

This is important for quickly loading batches of training data for higher
dimensional vision models.

The problem is related to rendering. For each voxel in some batch volume that
we're trying to assemble, we need to sample the relevant source data. That
sampling process can involve different data sources, different transformations,
etc.

If we're loading from a chunked data source like a zarr file, then this
amounts to figuring out how to hydrate and sample from chunks. Chunks are
are typically-compressesed and are supposed to be fast to randomly access.
I think it's best to decompress as late as possible (on the gpu) and to use
the gpu to handle transform and sampling.

I also think it might be useful to use some request coalescing strategy when
loading chunks from storage. This could be grouping reads that hit a single
shard, or, maybe more generally, just identifying when a run of chunk requests
resolve to a single contiguous byte request.

So those are some ideas.

