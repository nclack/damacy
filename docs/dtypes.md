# Output types

`BatchSpec.dtype` selects the type of the assembled batch on both CPU and GPU.
Integer outputs preserve every bit of same-type reads and every value in
lossless integer widening, including 64-bit values above float32's exact range.

```python
output = damacy.BatchSpec(samples=2, shape=(64, 256, 256), dtype="uint16")
```

| C enum / Python `Dtype` | Short name | Full name | Bytes per element |
| --- | --- | --- | ---: |
| `DAMACY_F32` / `F32` | `f32` | `float32` | 4 |
| `DAMACY_BF16` / `BF16` | `bf16` | `bfloat16` | 2 |
| `DAMACY_U8` / `U8` | `u8` | `uint8` | 1 |
| `DAMACY_U16` / `U16` | `u16` | `uint16` | 2 |
| `DAMACY_U32` / `U32` | `u32` | `uint32` | 4 |
| `DAMACY_U64` / `U64` | `u64` | `uint64` | 8 |
| `DAMACY_I8` / `I8` | `i8` | `int8` | 1 |
| `DAMACY_I16` / `I16` | `i16` | `int16` | 2 |
| `DAMACY_I32` / `I32` | `i32` | `int32` | 4 |
| `DAMACY_I64` / `I64` | `i64` | `int64` | 8 |

Python accepts both names, without regard to case, enum members, and their
integer values. The existing `F32=0` and `BF16=1` enum values are unchanged.
Benchmark scenarios accept short and full names in `pipeline.dtype`.

## Conversion rules

All eight integer source types and float16/float32 sources can produce any
output type in the table. CPU and GPU use the same rules, including for missing
chunks' fill values:

- Integer to integer: preserve values in range and clamp values outside the
  destination range. Negative values become zero for unsigned outputs.
  No integer-to-integer conversion passes through floating point.
- Float16/float32 to integer: truncate toward zero, then clamp to the destination
  range. NaN becomes zero. Positive infinity becomes the destination maximum;
  negative infinity becomes the minimum, which is zero for unsigned outputs.
- Integer or float16 to float32: round to the nearest representable float32
  value, with ties to even. Float32 sources retain their values.
- Bfloat16: first convert to float32, then round to bfloat16 with ties to even,
  retaining the existing floating-point output behavior.

For example, int16 values `[-1, 12, 256]` become uint8 values `[0, 12, 255]`.
Float32 values `[-1.9, 12.9, NaN, Infinity]` become int16 values
`[-1, 12, 0, 32767]`.

Float64 sources are unsupported for every output type. Planning reports
`DAMACY_DTYPE` (Python `DtypeMismatch`, raised by `pop`) before executing an
unsupported conversion. Invalid output types fail configuration validation.

## Buffers and consumers

Output buffers, memory budgets, and assembly byte counts use the selected
type's width. DLPack exports carry its signedness and bit width. NumPy can
consume CPU integer batches with `np.from_dlpack(batch)`; PyTorch can consume
CPU or CUDA integer batches with `torch.from_dlpack(batch)`. The consumer retains
the output buffer after the batch or pipeline is released.

A uint16 batch uses half the output memory of a float32 batch with the same
shape. Compare elapsed time or samples per second when benchmarking different
output widths: output GB/s alone counts different numbers of bytes.
