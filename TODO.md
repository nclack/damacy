# TODO

## To triage

- [ ] thread-safe public API (push/pop/flush)
- [x] support for fp16 sources
- [x] support for bf16 output
- [ ] testing the hashmap/lru
- [ ] add ngff and multiscale
- [ ] handle oob aabb's
- [x] type translation (e.g. u16 to bf16)
- [ ] eval lru for compressed chunk
      - may be interesting to eval hit rate
      - system's virtual page cache may obviate this for host mem
- [x] how to add back blosc support - add gpu support
- [ ] codegen for h100s
- [ ] comparative benchmark against other libs/solns: tensorstore, kvikio, DALI?

