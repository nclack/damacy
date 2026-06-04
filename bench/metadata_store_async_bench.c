#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "store/metadata_store_async.h"

#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum bench_mode
{
  MODE_READ_FILE,
  MODE_READ,
  MODE_STAT,
};

struct wait_state
{
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  uint64_t done;
  uint64_t errors;
  uint64_t bytes;
};

static double
now_seconds(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void
read_cb(void* user, enum damacy_status status, void* data, size_t len)
{
  struct wait_state* w = (struct wait_state*)user;
  pthread_mutex_lock(&w->mutex);
  w->done++;
  if (status != DAMACY_OK)
    w->errors++;
  else
    w->bytes += len;
  pthread_cond_signal(&w->cond);
  pthread_mutex_unlock(&w->mutex);
  free(data);
}

static void
stat_cb(void* user, enum store_stat_result status, uint64_t size)
{
  struct wait_state* w = (struct wait_state*)user;
  pthread_mutex_lock(&w->mutex);
  w->done++;
  if (status != STORE_STAT_OK)
    w->errors++;
  else
    w->bytes += size;
  pthread_cond_signal(&w->cond);
  pthread_mutex_unlock(&w->mutex);
}

static int
parse_u64(const char* s, uint64_t* out)
{
  char* end = NULL;
  errno = 0;
  unsigned long long v = strtoull(s, &end, 10);
  if (errno || !end || *end)
    return 1;
  *out = (uint64_t)v;
  return 0;
}

static int
parse_mode(const char* s, enum bench_mode* out)
{
  if (strcmp(s, "read-file") == 0) {
    *out = MODE_READ_FILE;
    return 0;
  }
  if (strcmp(s, "read") == 0) {
    *out = MODE_READ;
    return 0;
  }
  if (strcmp(s, "stat") == 0) {
    *out = MODE_STAT;
    return 0;
  }
  return 1;
}

static const char*
mode_name(enum bench_mode mode)
{
  switch (mode) {
    case MODE_READ_FILE:
      return "read-file";
    case MODE_READ:
      return "read";
    case MODE_STAT:
      return "stat";
  }
  return "unknown";
}

static void
usage(const char* argv0)
{
  fprintf(stderr,
          "usage: %s [--depths 1,2,4,8] [--requests N] [--files N] "
          "[--file-bytes N] [--read-bytes N] [--mode read-file|read|stat]\n",
          argv0);
}

static int
write_payload(const char* path, size_t nbytes)
{
  FILE* f = fopen(path, "wb");
  if (!f)
    return 1;
  unsigned char buf[4096];
  for (size_t i = 0; i < sizeof(buf); ++i)
    buf[i] = (unsigned char)(i & 0xffu);
  size_t left = nbytes;
  while (left) {
    size_t n = left < sizeof(buf) ? left : sizeof(buf);
    if (fwrite(buf, 1, n, f) != n) {
      fclose(f);
      return 1;
    }
    left -= n;
  }
  return fclose(f) != 0;
}

static char**
make_files(const char* root, uint64_t nfiles, size_t file_bytes)
{
  char** paths = (char**)calloc((size_t)nfiles, sizeof(*paths));
  if (!paths)
    return NULL;
  for (uint64_t i = 0; i < nfiles; ++i) {
    char tmp[512];
    snprintf(tmp, sizeof(tmp), "%s/file-%06llu", root, (unsigned long long)i);
    paths[i] = strdup(tmp);
    if (!paths[i] || write_payload(paths[i], file_bytes)) {
      for (uint64_t j = 0; j <= i; ++j) {
        if (paths[j])
          unlink(paths[j]);
        free(paths[j]);
      }
      free(paths);
      return NULL;
    }
  }
  return paths;
}

static void
remove_files(char** paths, uint64_t nfiles)
{
  if (!paths)
    return;
  for (uint64_t i = 0; i < nfiles; ++i) {
    if (paths[i])
      unlink(paths[i]);
    free(paths[i]);
  }
  free(paths);
}

static int
run_one(uint32_t depth,
        enum bench_mode mode,
        char** paths,
        uint64_t nfiles,
        uint64_t requests,
        size_t read_bytes)
{
  struct metadata_store_async* s =
    metadata_store_async_create(depth, NULL, NULL);
  if (!s) {
    fprintf(stderr, "failed to create metadata_store_async depth=%u\n", depth);
    return 1;
  }

  struct wait_state w = { 0 };
  pthread_mutex_init(&w.mutex, NULL);
  pthread_cond_init(&w.cond, NULL);

  double t0 = now_seconds();
  for (uint64_t i = 0; i < requests; ++i) {
    const char* path = paths[i % nfiles];
    int rc = 1;
    switch (mode) {
      case MODE_READ_FILE:
        rc = metadata_store_async_read_file(s, path, read_cb, &w);
        break;
      case MODE_READ:
        rc = metadata_store_async_read(s, path, 0, read_bytes, read_cb, &w);
        break;
      case MODE_STAT:
        rc = metadata_store_async_stat(s, path, stat_cb, &w);
        break;
    }
    if (rc) {
      pthread_mutex_lock(&w.mutex);
      w.done++;
      w.errors++;
      pthread_mutex_unlock(&w.mutex);
    }
  }

  pthread_mutex_lock(&w.mutex);
  while (w.done < requests)
    pthread_cond_wait(&w.cond, &w.mutex);
  pthread_mutex_unlock(&w.mutex);
  double elapsed = now_seconds() - t0;

  struct metadata_store_async_backend_stats st;
  metadata_store_async_backend_stats_get(s, &st);
  double req_s = elapsed > 0.0 ? (double)requests / elapsed : 0.0;
  double mib_s =
    elapsed > 0.0 ? ((double)w.bytes / (1024.0 * 1024.0)) / elapsed : 0.0;
  printf("%u,%s,%llu,%.6f,%.1f,%.1f,%llu,%llu,%llu,%llu\n",
         depth,
         mode_name(mode),
         (unsigned long long)requests,
         elapsed,
         req_s,
         mib_s,
         (unsigned long long)st.read_jobs,
         (unsigned long long)st.read_max_active,
         (unsigned long long)st.read_active,
         (unsigned long long)w.errors);

  metadata_store_async_destroy(s);
  pthread_cond_destroy(&w.cond);
  pthread_mutex_destroy(&w.mutex);
  return w.errors ? 1 : 0;
}

int
main(int argc, char** argv)
{
  const char* depths = "1,2,4,8,16,32,64";
  uint64_t requests = 20000;
  uint64_t nfiles = 1024;
  uint64_t file_bytes_u64 = 4096;
  uint64_t read_bytes_u64 = 256;
  enum bench_mode mode = MODE_READ_FILE;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--depths") == 0 && i + 1 < argc) {
      depths = argv[++i];
    } else if (strcmp(argv[i], "--requests") == 0 && i + 1 < argc) {
      if (parse_u64(argv[++i], &requests))
        goto BadArg;
    } else if (strcmp(argv[i], "--files") == 0 && i + 1 < argc) {
      if (parse_u64(argv[++i], &nfiles))
        goto BadArg;
    } else if (strcmp(argv[i], "--file-bytes") == 0 && i + 1 < argc) {
      if (parse_u64(argv[++i], &file_bytes_u64))
        goto BadArg;
    } else if (strcmp(argv[i], "--read-bytes") == 0 && i + 1 < argc) {
      if (parse_u64(argv[++i], &read_bytes_u64))
        goto BadArg;
    } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
      if (parse_mode(argv[++i], &mode))
        goto BadArg;
    } else {
      goto BadArg;
    }
  }

  if (!requests || !nfiles || !file_bytes_u64 || !read_bytes_u64)
    goto BadArg;
  if (file_bytes_u64 > SIZE_MAX || read_bytes_u64 > SIZE_MAX)
    goto BadArg;
  if (read_bytes_u64 > file_bytes_u64)
    goto BadArg;

  char root[] = "./.damacy_metadata_async_bench_XXXXXX";
  if (!mkdtemp(root)) {
    fprintf(stderr, "mkdtemp failed: %s\n", strerror(errno));
    return 1;
  }
  char** paths = make_files(root, nfiles, (size_t)file_bytes_u64);
  if (!paths) {
    fprintf(stderr, "failed to create benchmark files\n");
    rmdir(root);
    return 1;
  }

  printf("depth,mode,requests,seconds,requests_per_s,mib_per_s,"
         "read_jobs,read_max_active,read_active,errors\n");
  int rc = 0;
  char* depth_copy = strdup(depths);
  if (!depth_copy) {
    rc = 1;
    goto Done;
  }
  for (char* tok = strtok(depth_copy, ","); tok; tok = strtok(NULL, ",")) {
    uint64_t depth_u64 = 0;
    if (parse_u64(tok, &depth_u64) || depth_u64 == 0 ||
        depth_u64 > UINT32_MAX) {
      fprintf(stderr, "invalid depth: %s\n", tok);
      rc = 1;
      break;
    }
    if (run_one((uint32_t)depth_u64,
                mode,
                paths,
                nfiles,
                requests,
                (size_t)read_bytes_u64))
      rc = 1;
  }
  free(depth_copy);

Done:
  remove_files(paths, nfiles);
  rmdir(root);
  return rc;

BadArg:
  usage(argv[0]);
  return 2;
}
