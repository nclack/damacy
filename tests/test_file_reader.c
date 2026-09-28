#include "expect.h"
#include "pipeline/components.h"
#include "store/store_fs.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static char paths[2][128];

#ifdef __linux__
static unsigned advice_calls;
static int advice_error;
static int advised_fd;
static int invalid_advice;

int
__real_posix_fadvise(int fd, off_t offset, off_t len, int advice);

int
__wrap_posix_fadvise(int fd, off_t offset, off_t len, int advice)
{
  ++advice_calls;
  advised_fd = fd;
  invalid_advice |= offset != 0 || len != 0 || advice != POSIX_FADV_RANDOM;
  return advice_error ? advice_error
                      : __real_posix_fadvise(fd, offset, len, advice);
}
#endif

static int
read_byte(struct store_fs* fs, int index)
{
  struct lru_entry* pin = NULL;
  platform_file* file = store_fs_acquire(fs, paths[index], &pin);
  EXPECT(file);
  char value = 0;
  EXPECT(platform_file_pread(file, &value, 1, 0) == 1);
  EXPECT(value == 'a' + index);
  store_fs_release(fs, pin);
  return 0;
}

static int
test_reader_settings(void)
{
  struct damacy_reader* readers[3] = { 0 };
  EXPECT(damacy_file_reader_create(1, 4, &readers[0]) == DAMACY_OK);
  for (uint8_t enabled = 0; enabled <= 1; ++enabled) {
    const struct damacy_file_reader_config config = { .workers = 1,
                                                      .max_inflight_reads = 4,
                                                      .readahead = enabled };
    EXPECT(damacy_file_reader_create_with_config(
             &config, &readers[enabled + 1]) == DAMACY_OK);
  }
  for (int pass = 0; pass < 2; ++pass) {
    for (int index = 0; index < 2; ++index) {
      for (int reader = 0; reader < 3; ++reader) {
#ifdef __linux__
        unsigned before = advice_calls;
#endif
        EXPECT(read_byte((struct store_fs*)readers[reader]->store, index) == 0);
#ifdef __linux__
        EXPECT(advice_calls == before + (reader == 1 && pass == 0));
#endif
      }
    }
  }
  for (int reader = 0; reader < 3; ++reader)
    damacy_reader_destroy(readers[reader]);
  return 0;
}

static int
test_eviction_and_retry(void)
{
  const struct store_fs_config config = { .root = "",
                                          .nthreads = 1,
                                          .fd_cache_capacity = 1,
                                          .max_inflight_reads = 4,
                                          .disable_readahead = 1 };
  struct store* store = store_fs_create(&config);
  EXPECT(store);
  struct store_fs* fs = (struct store_fs*)store;
  for (int i = 0; i < 4; ++i) {
#ifdef __linux__
    unsigned before = advice_calls;
#endif
    EXPECT(read_byte(fs, i % 2) == 0);
#ifdef __linux__
    EXPECT(advice_calls == before + 1);
#endif
  }
#ifdef __linux__
  advice_error = EIO;
  struct lru_entry* pin = NULL;
  EXPECT(!store_fs_acquire(fs, paths[0], &pin));
  EXPECT(!pin);
  EXPECT(fcntl(advised_fd, F_GETFD) == -1 && errno == EBADF);
  advice_error = 0;
  unsigned before = advice_calls;
  EXPECT(read_byte(fs, 0) == 0);
  EXPECT(advice_calls == before + 1);
#endif
  store_destroy(store);
  return 0;
}

static int
test_invalid_config(void)
{
  struct damacy_reader* reader = NULL;
  struct damacy_file_reader_config config = { .workers = 1,
                                              .max_inflight_reads = 4,
                                              .readahead = 1 };
  EXPECT(damacy_file_reader_create_with_config(NULL, &reader) == DAMACY_INVAL);
  EXPECT(damacy_file_reader_create_with_config(&config, NULL) == DAMACY_INVAL);
  config.readahead = 2;
  EXPECT(damacy_file_reader_create_with_config(&config, &reader) ==
         DAMACY_INVAL);
  EXPECT(!reader);
  config.readahead = 1;
  config.workers = 0;
  EXPECT(damacy_file_reader_create_with_config(&config, &reader) ==
         DAMACY_INVAL);
  config.workers = 1;
  config.max_inflight_reads = 0;
  EXPECT(damacy_file_reader_create_with_config(&config, &reader) ==
         DAMACY_INVAL);
  EXPECT(platform_file_disable_readahead(NULL) == EINVAL);
  return 0;
}

int
main(void)
{
  for (int i = 0; i < 2; ++i) {
    snprintf(paths[i], sizeof(paths[i]), "/tmp/damacy_readahead_%d_XXXXXX", i);
    int fd = mkstemp(paths[i]);
    EXPECT(fd >= 0);
    char value = (char)('a' + i);
    EXPECT(write(fd, &value, 1) == 1);
    EXPECT(close(fd) == 0);
  }
  RUN(test_reader_settings);
  RUN(test_eviction_and_retry);
  RUN(test_invalid_config);
#ifdef __linux__
  EXPECT(!invalid_advice);
#endif
  for (int i = 0; i < 2; ++i)
    EXPECT(unlink(paths[i]) == 0);
  return 0;
}
