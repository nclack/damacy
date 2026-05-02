#include "platform/platform_io.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef O_DIRECT
#define O_DIRECT 0
#endif

struct platform_file
{
  int fd;
};

platform_file*
platform_file_open_read(const char* path, int o_direct)
{
  if (!path)
    return NULL;
  int flags = O_RDONLY;
#ifdef O_CLOEXEC
  flags |= O_CLOEXEC;
#endif
  if (o_direct)
    flags |= O_DIRECT;

  int fd = open(path, flags);
  if (fd < 0 && o_direct) {
    // Fall back to buffered I/O if the filesystem rejects O_DIRECT.
    flags &= ~O_DIRECT;
    fd = open(path, flags);
  }
  if (fd < 0)
    return NULL;

  platform_file* f = (platform_file*)calloc(1, sizeof(*f));
  if (!f) {
    close(fd);
    return NULL;
  }
  f->fd = fd;
  return f;
}

void
platform_file_close(platform_file* f)
{
  if (!f)
    return;
  if (f->fd >= 0)
    close(f->fd);
  free(f);
}

int64_t
platform_file_pread(platform_file* f, void* dst, size_t len, uint64_t offset)
{
  if (!f || !dst)
    return -1;

  uint8_t* p = (uint8_t*)dst;
  size_t left = len;
  uint64_t off = offset;
  while (left > 0) {
    ssize_t n = pread(f->fd, p, left, (off_t)off);
    if (n < 0) {
      if (errno == EINTR)
        continue;
      return -1;
    }
    if (n == 0)
      break; // EOF
    p += (size_t)n;
    left -= (size_t)n;
    off += (uint64_t)n;
  }
  return (int64_t)(len - left);
}

uint64_t
platform_file_size(platform_file* f)
{
  if (!f || f->fd < 0)
    return 0;
  struct stat st;
  if (fstat(f->fd, &st) != 0)
    return 0;
  return (uint64_t)st.st_size;
}

int
platform_path_size(const char* path, uint64_t* out)
{
  if (!path || !out)
    return 1;
  struct stat st;
  if (stat(path, &st) != 0)
    return 1;
  *out = (uint64_t)st.st_size;
  return 0;
}

int
platform_file_map_path(const char* path, struct platform_file_view* out)
{
  if (!path || !out)
    return 1;
  memset(out, 0, sizeof(*out));

  int flags = O_RDONLY;
#ifdef O_CLOEXEC
  flags |= O_CLOEXEC;
#endif
  int fd = open(path, flags);
  if (fd < 0)
    return 1;

  struct stat st;
  if (fstat(fd, &st) != 0) {
    close(fd);
    return 1;
  }
  size_t len = (size_t)st.st_size;
  if (len == 0) {
    close(fd);
    return 1;
  }

  void* p = mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
  // The mapping keeps its own reference to the file; the fd can be closed.
  close(fd);
  if (p == MAP_FAILED)
    return 1;

  out->data = p;
  out->len = len;
  out->opaque = NULL;
  return 0;
}

void
platform_file_unmap(struct platform_file_view* view)
{
  if (!view || !view->data || view->len == 0)
    return;
  munmap((void*)view->data, view->len);
  view->data = NULL;
  view->len = 0;
  view->opaque = NULL;
}
