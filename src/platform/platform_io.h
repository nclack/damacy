// Thin POSIX/Win32 wrapper for the read-side filesystem operations damacy
// needs. Currently:
//   - open a file for reading
//   - pread(fd, dst, len, offset) — required to be thread-safe per POSIX
//   - stat to recover file size
//   - close
//   - whole-file mmap for read-only consumers (metadata)
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

  // Read-only view into a mapped file. `data`/`len` are public; `opaque`
  // is backend-private (POSIX: unused; Win32 will stash the file mapping
  // handle here).
  struct platform_file_view
  {
    const void* data;
    size_t len;
    void* opaque;
  };

  // mmap `path` read-only over its full extent. Returns 0 on success and
  // populates *out. The view is independent of any open fd cache; the
  // mapping holds the file alive until platform_file_unmap.
  int platform_file_map_path(const char* path, struct platform_file_view* out);

  // Tear down a view returned by platform_file_map_path. Safe to call on
  // a zeroed view.
  void platform_file_unmap(struct platform_file_view* view);

#ifdef __cplusplus
}
#endif
