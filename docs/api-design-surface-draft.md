# damacy public C++ API — surface design

> Draft produced 2026-04-29 by the `surface` agent during the API design
> walkthrough. Parked while the C-language kvikio benchmark proceeds. This draft
> is C++-flavored; the eventual library is more likely to be Rust + pyo3 bindings
> (see project memory). This represents one independent angle on the public-facing
> API; some open questions and naming have evolved since (see appendix at the
> end).

## Header sketch

```cpp
// damacy/damacy.hpp
#pragma once
#include <array>
#include <cstdint>
#include <expected>
#include <generator>     // C++23
#include <memory>
#include <span>
#include <string_view>

namespace damacy {

enum class DType : uint8_t { u8, u16, i16, u32, f16, f32 };

// 5D shape over (t, c, z, y, x). Use 1 for unused leading dims.
struct Shape5 { int64_t t, c, z, y, x; };

// Half-open AABB in a source's index space at mip level 0.
// Spatial fields are inclusive-lo / exclusive-hi voxel indices;
// (t, c) are channel/time slabs.
struct AABB {
    std::array<int64_t, 5> lo;   // t, c, z, y, x
    std::array<int64_t, 5> hi;
};

struct SourceInfo {
    Shape5     shape_l0;     // shape at mip 0
    DType      dtype;
    int        n_levels;     // count of precomputed mips
    std::array<double, 3> voxel_size_um;  // z, y, x
};

// Opaque, ref-counted handle to an opened NGFF zarr store.
class Source {
public:
    SourceInfo info() const noexcept;
    std::string_view uri() const noexcept;
    // ...
private:
    struct Impl; std::shared_ptr<Impl> p_;
    friend std::expected<Source, class Error> open_source(std::string_view);
};

struct Error { int code; std::string message; };
template <class T> using Result = std::expected<T, Error>;

Result<Source> open_source(std::string_view uri);

// One sample request: a box in source space at a chosen mip.
struct SampleSpec {
    Source   source;
    AABB     aabb;        // expressed in level-0 indices
    int      mip = 0;     // which precomputed level to read
};

// What every sample in a batch shares.
struct BatchLayout {
    Shape5  sample_shape;   // shape of one sample after mip selection
    DType   dtype;
    int     batch_size;     // s
};

// GPU-resident batch. Library-owned via shared_ptr to a pool slot.
class Batch {
public:
    BatchLayout layout() const noexcept;
    void*       device_ptr() const noexcept;        // (s,t,c,z,y,x) contiguous
    size_t      nbytes()     const noexcept;
    int         device_id()  const noexcept;
    cudaStream_t ready_stream() const noexcept;     // wait on this before consuming

    // DLPack export. Capsule keeps the Batch alive via its deleter.
    DLManagedTensor* to_dlpack() &&;                // consumes the Batch
private:
    struct Impl; std::shared_ptr<Impl> p_;
    friend class BatchStream;
};

// --- Sampling strategies (the common case) ---------------------------------

struct UniformCropSampler {
    Source     source;
    Shape5     crop_shape;       // shape per sample, in voxels at mip
    int        mip = 0;
    uint64_t   seed = 0;
    // Optional: restrict where crops may originate (default: full volume).
    std::optional<AABB> within = std::nullopt;
};

// --- The stream ------------------------------------------------------------

struct StreamConfig {
    int batch_size      = 16;
    int prefetch_depth  = 2;     // batches in flight
    int device_id       = 0;
    size_t cache_bytes  = size_t(1) << 30;   // chunk-LRU budget
};

class BatchStream {
public:
    // High-level: one sampler, fixed batch size. Common case.
    template <class Sampler>
    static Result<BatchStream> make(Sampler s, StreamConfig cfg = {});

    // Low-level escape hatch: caller supplies each batch's specs.
    static Result<BatchStream> from_specs(
        std::function<std::vector<SampleSpec>(uint64_t batch_idx)> producer,
        StreamConfig cfg = {});

    // Pull-style. Blocks until the next batch is on-device-ready.
    Result<Batch> next();

    // Range-style sugar over next(). Infinite by default.
    std::generator<Batch> take(size_t n);

    void shutdown();
};

} // namespace damacy
```

## Training loop example

```cpp
#include <damacy/damacy.hpp>
using namespace damacy;

int main() {
    auto src = open_source("s3://bucket/embryo.ome.zarr").value();

    auto stream = BatchStream::make(
        UniformCropSampler{
            .source     = src,
            .crop_shape = {1, 2, 64, 128, 128},
            .mip        = 1,
            .seed       = 42,
        },
        StreamConfig{ .batch_size = 8, .prefetch_depth = 3 }
    ).value();

    for (Batch batch : stream.take(10'000)) {
        train_step(batch.device_ptr(), batch.layout(), batch.ready_stream());
    }
}
```

Python side:

```python
caps = damacy_cpp.next_batch_dlpack(stream)   # std::move(batch).to_dlpack()
x    = jax.dlpack.from_dlpack(caps)
```

## Rationale (most opinionated choices)

- **Sampler objects, not raw `(uri, aabb)` lists, for v1.** The common case is "uniform crops from one source." `UniformCropSampler` makes that 5 lines. `from_specs(producer)` is the escape hatch so the lower-level `SampleSpec` path stays first-class without polluting the easy path.
- **Library-owned `Batch` via pooled `shared_ptr`.** Caller-allocated buffers force the user to know `prefetch_depth * batch_nbytes` ahead of time and break the streaming pipeline's freedom to reuse slots. Pool + handle lets the runtime recycle GPU memory as soon as the user drops the `Batch`.
- **`std::expected<T, Error>` everywhere; no exceptions across the API.** Most consumers will be embedded in Python via pybind/nanobind; `expected` translates cleanly to Python exceptions at the boundary, and C++ training loops typically `.value()` or propagate. Exceptions through CUDA/coroutine code paths are a pit of despair.
- **`cudaStream_t ready_stream()` is exposed, not hidden.** The whole point of damacy is overlapping decode with training; the user must be able to make their training stream wait on ours without a host sync. This is the one piece of CUDA we refuse to abstract.
- **C++23 `std::generator` for iteration, not coroutines in the public API.** Range-for over `stream.take(N)` is what trainers actually want. `next()` stays available for non-iterating consumers (e.g., a Python `__next__`).

## Open questions

- AABB units: I chose level-0 voxels with a separate `mip` field. Alternative: AABB already in the chosen mip's units. Internals owner: which is cheaper to plumb through the chunk planner?
- DLPack stream semantics: do we expose our `ready_stream` via the DLPack v1.0 stream field, or require the Python caller to sync before handoff? Affects whether JAX/PyTorch zero-copy is truly async.
- Should `UniformCropSampler` accept a `mask` / `weights` source for foreground-biased sampling, or do we punt that to a `WeightedCropSampler` later? Punting feels right per "no hypothetical futures."
- `Shape5` vs. a variable-rank `Shape` template: NGFF is canonically 5D, but `(t=1, c=1)` everywhere is ugly. Worth a `Shape{c, z, y, x}` overload?
- Error taxonomy: is one `Error{code, message}` enough, or do we want `enum class ErrorKind { Io, Decode, Cuda, BadSpec, OutOfMemory }` for typed dispatch? I lean toward the enum for Python translation.

## User decisions during walkthrough (2026-04-29)

Some of the open questions were resolved during the walkthrough. Recording them here for when we come back to this draft:

- **AABB units: level-0 index space.** This draft already chose this — confirmed.
- **Mip selection: auto** from requested aabb resolution, not caller-specified. The `mip` field on `SampleSpec`/`UniformCropSampler` should become optional / a `target_voxel_size` style input. Goal: avoid aliasing and bound load cost to the sample size.
- **DLPack stream field: yes**, pass our `ready_stream` through so JAX/PyTorch consumers don't need a host sync.
- **GDS: deferred.** Host-pinned + cudaMemcpyAsync is the baseline. Don't design around GDS.
- **Read-config naming.** The GLSL-style "sampler" concept (interpolation/boundary/value-transform) needs a different name to avoid colliding with `Sampler`/`UniformCropSampler` (the training-side concept). Likely a per-axis config struct, where the axes include a "value axis" so dtype cast / affine / clip slot uniformly into the same shape. No LUT, no gamma. See project memory.
- **Implementation language: likely Rust + pyo3, not C++.** The damacy library proper will probably be Rust with Python bindings via `pyo3`/`maturin`. This draft's C++ headers are kept as a reference for the *public surface shape* — the eventual Rust API will mirror these types and verbs, just expressed as Rust.
