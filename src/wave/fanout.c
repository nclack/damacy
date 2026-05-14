#include "fanout.h"

#include "damacy_limits.h"
#include "gpu_budget/gpu_budget.h"
#include "log/log.h"
#include "util/cuda_check.h"
#include "util/prelude.h"

#include <stdint.h>
#include <stdlib.h>

int
fanout_alloc_pinned(struct blosc1_host_fanout* h,
                    struct nvcomp_fanout* d,
                    size_t n)
{
  CUdeviceptr dptr = 0;
  CU(Fail, cuMemAllocHost((void**)&h->comp_ptrs, n * sizeof(void*)));
  CU(Fail, cuMemAllocHost((void**)&h->comp_sizes, n * sizeof(size_t)));
  CU(Fail, cuMemAllocHost((void**)&h->decomp_ptrs, n * sizeof(void*)));
  CU(Fail, cuMemAllocHost((void**)&h->decomp_buf_sizes, n * sizeof(size_t)));
  CU(Fail, cuMemAlloc(&dptr, n * sizeof(void*)));
  d->d_comp_ptrs = (const void**)(uintptr_t)dptr;
  CU(Fail, cuMemAlloc(&dptr, n * sizeof(size_t)));
  d->d_comp_sizes = (size_t*)(uintptr_t)dptr;
  CU(Fail, cuMemAlloc(&dptr, n * sizeof(void*)));
  d->d_decomp_ptrs = (void**)(uintptr_t)dptr;
  CU(Fail, cuMemAlloc(&dptr, n * sizeof(size_t)));
  d->d_decomp_buf_sizes = (size_t*)(uintptr_t)dptr;
  return 0;
Fail:
  return 1;
}

void
fanout_free_pinned(struct blosc1_host_fanout* h, struct nvcomp_fanout* d)
{
  if (h) {
    void* host_ptrs[] = {
      (void*)h->comp_ptrs,
      h->comp_sizes,
      h->decomp_ptrs,
      h->decomp_buf_sizes,
    };
    for (size_t i = 0; i < countof(host_ptrs); ++i)
      if (host_ptrs[i])
        cuMemFreeHost(host_ptrs[i]);
    h->comp_ptrs = NULL;
    h->comp_sizes = NULL;
    h->decomp_ptrs = NULL;
    h->decomp_buf_sizes = NULL;
  }
  if (d) {
    void* dev_ptrs[] = {
      (void*)d->d_comp_ptrs,
      d->d_comp_sizes,
      d->d_decomp_ptrs,
      d->d_decomp_buf_sizes,
    };
    for (size_t i = 0; i < countof(dev_ptrs); ++i)
      if (dev_ptrs[i])
        cuMemFree(CUDPTR(dev_ptrs[i]));
    d->d_comp_ptrs = NULL;
    d->d_comp_sizes = NULL;
    d->d_decomp_ptrs = NULL;
    d->d_decomp_buf_sizes = NULL;
  }
}

enum damacy_status
fanout_upload(CUstream s,
              const struct nvcomp_fanout* d,
              const struct blosc1_host_fanout* h,
              size_t n)
{
  CU(Fail,
     cuMemcpyHtoDAsync(
       CUDPTR(d->d_comp_ptrs), h->comp_ptrs, n * sizeof(void*), s));
  CU(Fail,
     cuMemcpyHtoDAsync(
       CUDPTR(d->d_comp_sizes), h->comp_sizes, n * sizeof(size_t), s));
  CU(Fail,
     cuMemcpyHtoDAsync(
       CUDPTR(d->d_decomp_ptrs), h->decomp_ptrs, n * sizeof(void*), s));
  CU(Fail,
     cuMemcpyHtoDAsync(CUDPTR(d->d_decomp_buf_sizes),
                       h->decomp_buf_sizes,
                       n * sizeof(size_t),
                       s));
  return DAMACY_OK;
Fail:
  return DAMACY_CUDA;
}

size_t
fanout_next_pow2(size_t v)
{
  if (v <= 1)
    return 1;
  --v;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
#if SIZE_MAX > 0xFFFFFFFFu
  v |= v >> 32;
#endif
  return v + 1;
}

// Device-resident bytes per substream in a fanout SOA: 2 pointers and
// 2 size_t (mirrors fanout_alloc_pinned). Pinned-host fanout bytes
// don't count against the GPU budget.
static uint64_t
fanout_dev_bytes_per_sub(void)
{
  return (uint64_t)(2 * sizeof(void*) + 2 * sizeof(size_t));
}

enum damacy_status
fanout_grow(struct blosc1_host_fanout* h,
            struct nvcomp_fanout* d,
            uint32_t* cap,
            size_t need,
            struct gpu_budget* budget)
{
  if (need <= (size_t)*cap)
    return DAMACY_OK;
  size_t new_cap = fanout_next_pow2(need);
  if (new_cap > (size_t)DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE)
    new_cap = (size_t)DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE;

  const uint32_t cur = *cap;
  const uint64_t per_sub = fanout_dev_bytes_per_sub();
  const uint64_t old_bytes = (uint64_t)cur * per_sub;
  const uint64_t new_bytes = (uint64_t)new_cap * per_sub;
  const uint64_t delta_bytes = new_bytes - old_bytes;
  // Reject before freeing the existing fanout so the wave stays usable
  // if the grow would breach the budget.
  enum damacy_status bs =
    gpu_budget_try_commit(budget, delta_bytes, "wave fanout grow");
  if (bs != DAMACY_OK)
    return bs;

  fanout_free_pinned(h, d);
  *cap = 0;
  if (fanout_alloc_pinned(h, d, new_cap) != 0) {
    // The pre-grow allocation is gone too; release the full new_bytes
    // so committed reflects an empty fanout (matches *cap == 0).
    gpu_budget_release(budget, new_bytes);
    return DAMACY_CUDA;
  }
  *cap = (uint32_t)new_cap;
  log_info("wave fanout: grew %u -> %zu (need=%zu, +%llu bytes)",
           (unsigned)cur,
           new_cap,
           need,
           (unsigned long long)delta_bytes);
  return DAMACY_OK;
}
