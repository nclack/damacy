// Thin POSIX/Win32 wrapper for the read-side filesystem operations damacy
// needs. Currently:
//   - open a file for reading
//   - pread(fd, dst, len, offset) — required to be thread-safe per POSIX
//   - stat to recover file size
//   - close
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Opaque file handle.
  typedef struct platform_file platform_file;

  // Open `path` for reading. If `o_direct` is non-zero, attempt O_DIRECT
  // (advisory; the wrapper falls back to buffered I/O on systems where
  // O_DIRECT is unsupported for this path). Returns NULL on error.
  platform_file* platform_file_open_read(const char* path, int o_direct);

  void platform_file_close(platform_file* f);

  // Positional read. Returns bytes read on success (== len when not EOF),
  // -1 on error. Thread-safe per POSIX semantics.
  int64_t platform_file_pread(platform_file* f,
                              void* dst,
                              size_t len,
                              uint64_t offset);

  // File size in bytes. Returns 0 on error.
  uint64_t platform_file_size(platform_file* f);

  // Stat-by-path convenience. Returns 0 on success, non-zero on error.
  int platform_path_size(const char* path, uint64_t* out);

#ifdef __cplusplus
}
#endif
