#include "log.h"

#include <stdarg.h>
#include <stdio.h>
#include <time.h>

#define LOG_MSG_BUFFER 2048

static const char* level_strings[] = {
  "TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL",
};

static int g_level = LOG_INFO;

void
log_set_level(int level)
{
  g_level = level;
}

static void
fmt_time(char* out, size_t cap, const struct timespec* ts)
{
  struct tm tm;
  time_t sec = ts->tv_sec;
#ifdef _WIN32
  localtime_s(&tm, &sec);
#else
  localtime_r(&sec, &tm);
#endif
  char hms[16];
  strftime(hms, sizeof hms, "%H:%M:%S", &tm);
  snprintf(out, cap, "%s.%03ld", hms, (long)(ts->tv_nsec / 1000000L));
}

void
log_log(int level, const char* file, int line, const char* fmt, ...)
{
  if (level < g_level)
    return;

  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  char timebuf[32];
  fmt_time(timebuf, sizeof timebuf, &ts);

  char msg[LOG_MSG_BUFFER];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(msg, sizeof msg, fmt, ap);
  va_end(ap);

  fprintf(stderr,
          "%s %-5s %s:%d: %s\n",
          timebuf,
          level_strings[level],
          file,
          line,
          msg);
  fflush(stderr);
}
