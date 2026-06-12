// Mount-ceiling probe: replays a captured damacy read-op stream with
// nothing but pread workers, so its throughput is the best any pipeline
// could do for that op stream on this mount at this queue depth.
//
// Capture a trace from any damacy run with DAMACY_TRACE_READS=<file>
// (one line per op, "path offset len", in submission order). Replay:
//
//   preadreplay pinned direct <trace> 64
//
// `direct` (O_DIRECT) bypasses the page cache — required for a cold
// ceiling on a node that has already read the data. `pinned` uses CUDA
// page-locked buffers like damacy's staging slots. Compare the bench
// summary's "wire rate" against this number; their ratio is pipeline
// efficiency, and ceiling / "read amplification" is the expected peak
// throughput.
//
// N threads pop ops from a shared atomic cursor in file order —
// mirrors damacy's io_queue FIFO + blocking-pread workers. Trace ops
// are 4096-aligned (the planner pads reads); a short read is legal only
// when the op tail crosses EOF.
#define _GNU_SOURCE
#include <cuda.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define MAX_THREADS 256
#define HASH_SLOTS (1u << 13)

struct op
{
  int fi;
  uint64_t off;
  uint64_t len;
};

struct file_ent
{
  char* path;
  int fd;
  off_t size;
};

static struct file_ent g_files[HASH_SLOTS];
static int g_nfiles;
static struct op* g_ops;
static long g_nops;
static uint64_t* g_lat_ns; // per-op, indexed in trace order
static atomic_long g_next;
static atomic_ulong g_bytes;
static atomic_int g_fail;
static char* g_bufs[MAX_THREADS];
static int g_open_flags;

static uint32_t
hash_str(const char* s)
{
  uint32_t h = 2166136261u;
  for (; *s; ++s)
    h = (h ^ (uint8_t)*s) * 16777619u;
  return h;
}

static int
file_idx(const char* path)
{
  if (g_nfiles >= (int)(HASH_SLOTS / 2))
    return fprintf(stderr, "too many distinct files\n"), -1;
  uint32_t slot = hash_str(path) & (HASH_SLOTS - 1);
  for (;;) {
    if (!g_files[slot].path) {
      int fd = open(path, g_open_flags);
      if (fd < 0)
        return perror(path), -1;
      g_files[slot].path = strdup(path);
      g_files[slot].fd = fd;
      g_files[slot].size = lseek(fd, 0, SEEK_END);
      g_nfiles++;
      return (int)slot;
    }
    if (strcmp(g_files[slot].path, path) == 0)
      return (int)slot;
    slot = (slot + 1) & (HASH_SLOTS - 1);
  }
}

static uint64_t
now_ns(void)
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static void*
worker(void* arg)
{
  char* buf = g_bufs[(uintptr_t)arg];
  for (;;) {
    long i = atomic_fetch_add(&g_next, 1);
    if (i >= g_nops || atomic_load(&g_fail))
      break;
    struct op* o = &g_ops[i];
    struct file_ent* f = &g_files[o->fi];
    uint64_t t0 = now_ns();
    ssize_t n = pread(f->fd, buf, o->len, (off_t)o->off);
    g_lat_ns[i] = now_ns() - t0;
    if (n < 0) {
      perror("pread");
      atomic_store(&g_fail, 1);
      break;
    }
    if ((uint64_t)n < o->len && o->off + (uint64_t)n < (uint64_t)f->size) {
      fprintf(stderr,
              "short read mid-file: %s off=%llu len=%llu got=%zd\n",
              f->path,
              (unsigned long long)o->off,
              (unsigned long long)o->len,
              n);
      atomic_store(&g_fail, 1);
      break;
    }
    atomic_fetch_add(&g_bytes, (unsigned long)n);
  }
  return NULL;
}

static int
cmp_u64(const void* a, const void* b)
{
  uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b;
  return x < y ? -1 : x > y;
}

int
main(int argc, char** argv)
{
  if (argc < 4 || argc > 5) {
    fprintf(stderr,
            "usage: %s malloc|pinned direct|buffered trace [nthreads]\n",
            argv[0]);
    return 1;
  }
  int pinned = strcmp(argv[1], "pinned") == 0;
  int direct = strcmp(argv[2], "direct") == 0;
  int nthreads = argc == 5 ? atoi(argv[4]) : 64;
  if (nthreads < 1 || nthreads > MAX_THREADS)
    return fprintf(stderr, "bad nthreads\n"), 1;
  g_open_flags = O_RDONLY | (direct ? O_DIRECT : 0);

  FILE* tf = fopen(argv[3], "r");
  if (!tf)
    return perror(argv[3]), 1;
  long cap = 1 << 16;
  g_ops = malloc(cap * sizeof(*g_ops));
  char line[4096];
  uint64_t max_len = 0;
  while (fgets(line, sizeof line, tf)) {
    char path[4096];
    unsigned long long off, len;
    if (sscanf(line, "%4095s %llu %llu", path, &off, &len) != 3)
      continue;
    if (g_nops == cap)
      g_ops = realloc(g_ops, (cap *= 2) * sizeof(*g_ops));
    int fi = file_idx(path);
    if (fi < 0)
      return 1;
    if (direct && ((off | len) & 4095))
      return fprintf(stderr, "unaligned op: %s", line), 1;
    g_ops[g_nops++] = (struct op){ .fi = fi, .off = off, .len = len };
    if (len > max_len)
      max_len = len;
  }
  fclose(tf);
  if (!g_nops)
    return fprintf(stderr, "empty trace\n"), 1;
  g_lat_ns = calloc(g_nops, sizeof(*g_lat_ns));

  if (pinned) {
    if (cuInit(0) != CUDA_SUCCESS)
      return fprintf(stderr, "cuInit failed\n"), 1;
    CUdevice dev;
    CUcontext ctx;
    if (cuDeviceGet(&dev, 0) || cuDevicePrimaryCtxRetain(&ctx, dev) ||
        cuCtxSetCurrent(ctx))
      return fprintf(stderr, "cuda ctx failed\n"), 1;
    void* big = NULL;
    if (cuMemAllocHost(&big, (size_t)nthreads * max_len) != CUDA_SUCCESS)
      return fprintf(stderr, "cuMemAllocHost failed\n"), 1;
    for (int t = 0; t < nthreads; ++t)
      g_bufs[t] = (char*)big + (size_t)t * max_len;
  } else {
    for (int t = 0; t < nthreads; ++t)
      if (posix_memalign((void**)&g_bufs[t], 4096, max_len))
        return fprintf(stderr, "memalign failed\n"), 1;
  }

  pthread_t th[MAX_THREADS];
  uint64_t t0 = now_ns();
  for (long t = 0; t < nthreads; ++t)
    pthread_create(&th[t], NULL, worker, (void*)t);
  for (int t = 0; t < nthreads; ++t)
    pthread_join(th[t], NULL);
  double dt = (double)(now_ns() - t0) / 1e9;
  if (atomic_load(&g_fail))
    return fprintf(stderr, "FAILED\n"), 1;

  double gb = (double)atomic_load(&g_bytes) / 1e9;
  qsort(g_lat_ns, g_nops, sizeof(uint64_t), cmp_u64);
  double mean = 0;
  for (long i = 0; i < g_nops; ++i)
    mean += (double)g_lat_ns[i];
  mean /= (double)g_nops * 1e6;
  printf("%-6s %-8s nthreads=%-3d nops=%-6ld nfiles=%-4d "
         "%.2f GB in %.2fs = %.2f GB/s | lat ms mean=%.2f p50=%.2f "
         "p90=%.2f p99=%.2f max=%.1f\n",
         argv[1],
         argv[2],
         nthreads,
         g_nops,
         g_nfiles,
         gb,
         dt,
         gb / dt,
         mean,
         (double)g_lat_ns[g_nops / 2] / 1e6,
         (double)g_lat_ns[(long)((double)g_nops * 0.90)] / 1e6,
         (double)g_lat_ns[(long)((double)g_nops * 0.99)] / 1e6,
         (double)g_lat_ns[g_nops - 1] / 1e6);
  return 0;
}
