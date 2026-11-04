#include "cold_read.h"
#include "util/json.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __linux__
#include <fcntl.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

struct cached_file
{
  int fd;
  void* mapping;
  size_t length;
  size_t pages;
  unsigned char* resident;
};

struct device_counter
{
  int fd;
  struct cold_read_counts before;
};

struct cold_read
{
  struct cached_file* files;
  size_t nfiles;
  struct device_counter* devices;
  size_t ndevices;
  struct cold_read_result result;
};

static double
seconds(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec / 1e9;
}

static int
open_paths(struct cslice source, const char* key, struct cold_read* state)
{
  struct json_query query[] = { { QUERY_KEY, .key = "cache" },
                                { QUERY_KEY, .key = key },
                                { .kind = QUERY_ITER } };
  struct json_iter it;
  if (json_iter_init(source, query, 3, &it, NULL))
    return 1;
  for (;;) {
    struct json_node value;
    enum json_err error = json_iter_next(&it, &value);
    if (error == JSON_ERR_NOT_FOUND)
      break;
    if (error != JSON_OK || value.type != JSON_STRING ||
        value.flag == JSON_NODE_FLAG_STR_ESCAPED)
      return 1;
    char* path = strndup(value.s.beg, cslice_len(value.s));
    if (!path)
      return 1;
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0)
      fprintf(stderr, "bench: cannot open %s: %s\n", path, strerror(errno));
    free(path);
    if (fd < 0)
      return 1;
    if (!strcmp(key, "device_stats")) {
      struct device_counter* devices =
        realloc(state->devices, (state->ndevices + 1) * sizeof(*devices));
      if (!devices) {
        close(fd);
        return 1;
      }
      state->devices = devices;
      state->devices[state->ndevices++] = (struct device_counter){ .fd = fd };
    } else {
      struct stat status;
      if (fstat(fd, &status) || !S_ISREG(status.st_mode) ||
          status.st_uid != geteuid() || status.st_size <= 0 || fsync(fd)) {
        fprintf(stderr,
                "bench: cold reads require owned, flushed regular files\n");
        close(fd);
        return 1;
      }
      struct cached_file* files =
        realloc(state->files, (state->nfiles + 1) * sizeof(*files));
      if (!files) {
        close(fd);
        return 1;
      }
      state->files = files;
      struct cached_file* file = &state->files[state->nfiles++];
      size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
      *file = (struct cached_file){
        .fd = fd,
        .length = (size_t)status.st_size,
        .pages = ((size_t)status.st_size + page_size - 1) / page_size
      };
      file->mapping = mmap(NULL, file->length, PROT_NONE, MAP_SHARED, fd, 0);
      file->resident = malloc(file->pages);
      if (file->mapping == MAP_FAILED || !file->resident)
        return 1;
      state->result.pages += file->pages;
    }
  }
  return 0;
}

static int
read_counter(int fd, struct cold_read_counts* out)
{
  char buffer[1024];
  ssize_t n = pread(fd, buffer, sizeof(buffer) - 1, 0);
  if (n <= 0)
    return 1;
  buffer[n] = '\0';
  uint64_t sectors, merged, read_ms;
  if (sscanf(buffer,
             "%" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64,
             &out->reads,
             &merged,
             &sectors,
             &read_ms,
             &out->writes) != 5)
    return 1;
  out->bytes = sectors * 512;
  return 0;
}

static int
snapshot(struct cold_read* state)
{
  for (size_t i = 0; i < state->ndevices; ++i)
    if (read_counter(state->devices[i].fd, &state->devices[i].before))
      return 1;
  return 0;
}

static int
difference(struct cold_read* state, struct cold_read_counts* out)
{
  for (size_t i = 0; i < state->ndevices; ++i) {
    struct cold_read_counts after, before = state->devices[i].before;
    if (read_counter(state->devices[i].fd, &after) ||
        after.reads < before.reads || after.bytes < before.bytes ||
        after.writes < before.writes)
      return 1;
    out->reads += after.reads - before.reads;
    out->bytes += after.bytes - before.bytes;
    out->writes += after.writes - before.writes;
  }
  return 0;
}

static int
resident_pages(struct cached_file* file, uint64_t* out)
{
  if (mincore(file->mapping, file->length, file->resident))
    return 1;
  *out = 0;
  for (size_t i = 0; i < file->pages; ++i)
    *out += file->resident[i] & 1;
  return 0;
}

int
cold_read_create(struct cslice source, struct cold_read** out)
{
  struct cold_read* state = calloc(1, sizeof(*state));
  if (!state)
    return 1;
  if (open_paths(source, "files", state) ||
      open_paths(source, "device_stats", state) || !state->nfiles ||
      !state->ndevices || snapshot(state)) {
    cold_read_destroy(state);
    return 1;
  }
  state->result.files = state->nfiles;
  *out = state;
  return 0;
}

int
cold_read_begin(struct cold_read* state)
{
  double start = seconds();
  for (size_t i = 0; i < state->nfiles; ++i) {
    struct cached_file* file = &state->files[i];
    uint64_t resident;
    if (resident_pages(file, &resident))
      return 1;
    state->result.resident_pages_before += resident;
    for (unsigned attempt = 0; attempt < 10; ++attempt) {
      int error = posix_fadvise(file->fd, 0, 0, POSIX_FADV_DONTNEED);
      if (error || resident_pages(file, &resident))
        return 1;
      if (!resident)
        break;
      struct timespec delay = { .tv_nsec = 1000000 };
      nanosleep(&delay, NULL);
    }
    if (resident) {
      fprintf(stderr, "bench: file pages remained cached; cold run rejected\n");
      return 1;
    }
    state->result.resident_pages_after += resident;
  }
  state->result.preparation_ms += (seconds() - start) * 1e3;
  return snapshot(state);
}

int
cold_read_end(struct cold_read* state)
{
  struct cold_read_counts delta = { 0 };
  if (difference(state, &delta) || !delta.reads || !delta.bytes) {
    fprintf(stderr, "bench: no device reads observed; cold run rejected\n");
    return 1;
  }
  state->result.device.reads += delta.reads;
  state->result.device.bytes += delta.bytes;
  state->result.device.writes += delta.writes;
  ++state->result.passes;
  return 0;
}

int
cold_read_idle(struct cold_read* state)
{
  if (snapshot(state))
    return 1;
  double start = seconds();
  struct timespec delay = { .tv_nsec = 100000000 };
  while (nanosleep(&delay, &delay) && errno == EINTR) {
  }
  state->result.idle_seconds += seconds() - start;
  return difference(state, &state->result.idle);
}

void
cold_read_result(struct cold_read* state, struct cold_read_result* out)
{
  *out = state->result;
}

void
cold_read_destroy(struct cold_read* state)
{
  if (!state)
    return;
  for (size_t i = 0; i < state->nfiles; ++i) {
    close(state->files[i].fd);
    if (state->files[i].mapping != MAP_FAILED)
      munmap(state->files[i].mapping, state->files[i].length);
    free(state->files[i].resident);
  }
  for (size_t i = 0; i < state->ndevices; ++i)
    close(state->devices[i].fd);
  free(state->files);
  free(state->devices);
  free(state);
}
#else
int
cold_read_create(struct cslice source, struct cold_read** out)
{
  (void)source;
  (void)out;
  fprintf(stderr, "bench: cold read verification requires Linux\n");
  return 1;
}
int
cold_read_begin(struct cold_read* state)
{
  (void)state;
  return 1;
}
int
cold_read_end(struct cold_read* state)
{
  (void)state;
  return 1;
}
int
cold_read_idle(struct cold_read* state)
{
  (void)state;
  return 1;
}
void
cold_read_result(struct cold_read* state, struct cold_read_result* out)
{
  (void)state;
  (void)out;
}
void
cold_read_destroy(struct cold_read* state)
{
  (void)state;
}
#endif
