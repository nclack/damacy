#include "log/log.h"
#include "damacy_log.h"
#include "platform/platform.h"

#include <stdarg.h>
#include <stdio.h>

#define DAMACY_LOG_MAX_CALLBACKS 32
#define DAMACY_LOG_MSG_BUFFER 2048

struct callback
{
  damacy_log_fn fn;
  void* udata;
  damacy_log_level threshold;
  int in_use;
};

static struct
{
  damacy_log_level level;
  int quiet;
  int truncation_warned;
  struct callback callbacks[DAMACY_LOG_MAX_CALLBACKS];
} L = { .level = DAMACY_LOG_INFO };

static const char* level_strings[] = {
  "TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL",
};

static void
stderr_sink(const damacy_log_event* ev, void* udata)
{
  (void)udata;
  struct tm tm;
  platform_localtime(&ev->time.tv_sec, &tm);
  char timebuf[16];
  strftime(timebuf, sizeof(timebuf), "%H:%M:%S", &tm);
  fprintf(stderr,
          "%s.%03ld %-5s %s:%d: %s\n",
          timebuf,
          (long)(ev->time.tv_nsec / 1000000L),
          level_strings[ev->level],
          ev->file,
          ev->line,
          ev->msg);
  fflush(stderr);
}

void
damacy_log_set_level(damacy_log_level level)
{
  L.level = level;
}

void
damacy_log_set_quiet(int quiet)
{
  L.quiet = quiet != 0;
}

int
damacy_log_add_callback(damacy_log_fn fn,
                        void* udata,
                        damacy_log_level threshold)
{
  if (!fn)
    return -1;
  for (int i = 0; i < DAMACY_LOG_MAX_CALLBACKS; i++) {
    if (!L.callbacks[i].in_use) {
      L.callbacks[i].fn = fn;
      L.callbacks[i].udata = udata;
      L.callbacks[i].threshold = threshold;
      L.callbacks[i].in_use = 1;
      return 0;
    }
  }
  return -1;
}

// Lowest level any active sink will accept. Used for the fast-path early
// return so we don't format messages no one will read.
static int
min_active_threshold(void)
{
  int floor = DAMACY_LOG_FATAL + 1;
  if (!L.quiet && (int)L.level < floor)
    floor = (int)L.level;
  for (int i = 0; i < DAMACY_LOG_MAX_CALLBACKS; i++) {
    if (L.callbacks[i].in_use && (int)L.callbacks[i].threshold < floor)
      floor = (int)L.callbacks[i].threshold;
  }
  return floor;
}

void
log_log(int level, const char* file, int line, const char* fmt, ...)
{
  if (level < min_active_threshold())
    return;

  char buf[DAMACY_LOG_MSG_BUFFER];
  va_list ap;
  va_start(ap, fmt);
  int needed = vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);

  damacy_log_event ev = {
    .msg = buf,
    .file = file,
    .line = line,
    .level = (damacy_log_level)level,
  };
  timespec_get(&ev.time, TIME_UTC);

  if (!L.quiet && level >= (int)L.level)
    stderr_sink(&ev, NULL);

  for (int i = 0; i < DAMACY_LOG_MAX_CALLBACKS; i++) {
    const struct callback* cb = &L.callbacks[i];
    if (cb->in_use && ev.level >= cb->threshold)
      cb->fn(&ev, cb->udata);
  }

  // First-time truncation notice. Recursion is bounded: the warning
  // message is short, the flag is set before recursing, and
  // L.truncation_warned gates any subsequent entry.
  if (needed >= (int)sizeof(buf) && !L.truncation_warned) {
    L.truncation_warned = 1;
    log_warn("damacy_log: message truncated at %d bytes "
             "(increase DAMACY_LOG_MSG_BUFFER)",
             (int)sizeof(buf));
  }
}
