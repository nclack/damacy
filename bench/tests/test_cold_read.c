#include "cold_read.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define EXPECT(expression)                                                     \
  do {                                                                         \
    if (!(expression)) {                                                       \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expression);         \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static int
counter(int fd, const char* value)
{
  size_t length = strlen(value);
  return ftruncate(fd, 0) || pwrite(fd, value, length, 0) != (ssize_t)length;
}

int
main(void)
{
  char root[] = "/tmp/damacy-cache-test.XXXXXX";
  EXPECT(mkdtemp(root) != NULL);
  char data_path[256], stat_path[256], json[1024];
  snprintf(data_path, sizeof(data_path), "%s/data", root);
  snprintf(stat_path, sizeof(stat_path), "%s/counters", root);
  int data = open(data_path, O_CREAT | O_EXCL | O_RDWR, 0600);
  int stats = open(stat_path, O_CREAT | O_EXCL | O_RDWR, 0600);
  EXPECT(data >= 0 && stats >= 0);
  char bytes[65536];
  memset(bytes, 37, sizeof(bytes));
  EXPECT(write(data, bytes, sizeof(bytes)) == sizeof(bytes));
  EXPECT(counter(stats, "10 0 128 0 1\n") == 0);
  snprintf(json,
           sizeof(json),
           "{\"cache\":{\"files\":[\"%s\"],\"device_stats\":[\"%s\"]}}",
           data_path,
           stat_path);
  struct cslice source = { .beg = json, .end = json + strlen(json) };
  struct cold_read* state = NULL;
  EXPECT(cold_read_create(source, &state) == 0);
  EXPECT(cold_read_begin(state) == 0);
  EXPECT(pread(data, bytes, sizeof(bytes), 0) == sizeof(bytes));
  EXPECT(bytes[0] == 37 && bytes[sizeof(bytes) - 1] == 37);
  EXPECT(counter(stats, "12 0 256 0 1\n") == 0);
  EXPECT(cold_read_end(state) == 0);
  EXPECT(cold_read_begin(state) == 0);
  EXPECT(pread(data, bytes, sizeof(bytes), 0) == sizeof(bytes));
  EXPECT(counter(stats, "15 0 384 0 2\n") == 0);
  EXPECT(cold_read_end(state) == 0);
  EXPECT(cold_read_idle(state) == 0);
  struct cold_read_result result;
  cold_read_result(state, &result);
  EXPECT(result.passes == 2 && result.files == 1);
  EXPECT(result.resident_pages_before > 0);
  EXPECT(result.resident_pages_after == 0);
  EXPECT(result.device.reads == 5 && result.device.bytes == 131072);
  EXPECT(result.device.writes == 1 && result.idle.reads == 0);
  EXPECT(result.idle_seconds >= 0.09 && result.preparation_ms > 0);

  EXPECT(cold_read_begin(state) == 0);
  EXPECT(cold_read_end(state) != 0);
  EXPECT(counter(stats, "1 0 1 0 0\n") == 0);
  EXPECT(cold_read_end(state) != 0);
  void* mapping = mmap(NULL, sizeof(bytes), PROT_READ, MAP_SHARED, data, 0);
  EXPECT(mapping != MAP_FAILED);
  EXPECT(mlock(mapping, sizeof(bytes)) == 0);
  EXPECT(cold_read_begin(state) != 0);
  EXPECT(munlock(mapping, sizeof(bytes)) == 0);
  EXPECT(munmap(mapping, sizeof(bytes)) == 0);
  cold_read_destroy(state);

  const char empty[] = "{\"cache\":{\"files\":[],\"device_stats\":[]}}";
  EXPECT(cold_read_create(
           (struct cslice){ .beg = empty, .end = empty + strlen(empty) },
           &state) != 0);
  close(data);
  close(stats);
  EXPECT(unlink(data_path) == 0);
  EXPECT(unlink(stat_path) == 0);
  EXPECT(rmdir(root) == 0);
  return 0;
}
