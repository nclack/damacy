// NVTX wrapper implementation. The body is only compiled in when
// DAMACY_NVTX_ENABLED=1; otherwise the file is a tiny TU that exists
// just so damacy_nvtx has a linker language. The disabled flavor's
// call sites land on the inline no-ops in nvtx.h, so there's nothing
// to define here.

#include "nvtx/nvtx.h"

#if DAMACY_NVTX_ENABLED

#include <stdarg.h>
#include <stdio.h>

#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCuda.h>

#include "platform/platform.h"

// Per-thread printf buffer for damacy_nvtx_range_pushf. Range names are
// short labels like "kick_h2d/w0" or "finalize_wave/w1" (well under 32
// chars); 128 is generous headroom. vsnprintf truncates silently on
// overflow, which is acceptable for a profiling string — worst case is
// a slightly less informative timeline entry.
#define DAMACY_NVTX_FMT_BUF 128

// Lazily-created via platform_call_once so concurrent first-callers
// can't race on the domain handle (plain double-check on a non-atomic
// static pointer is C11 UB and leaks one of the two domains).
static platform_once s_domain_once = PLATFORM_ONCE_INIT;
static nvtxDomainHandle_t s_domain = NULL;

static void
init_domain(void)
{
  s_domain = nvtxDomainCreateA("damacy");
}

static nvtxDomainHandle_t
get_domain(void)
{
  platform_call_once(&s_domain_once, init_domain);
  return s_domain;
}

void
damacy_nvtx_range_push(const char* name)
{
  if (!name)
    name = "(null)";
  nvtxEventAttributes_t ev = { 0 };
  ev.version = NVTX_VERSION;
  ev.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  ev.messageType = NVTX_MESSAGE_TYPE_ASCII;
  ev.message.ascii = name;
  nvtxDomainRangePushEx(get_domain(), &ev);
}

void
damacy_nvtx_range_pushf(const char* fmt, ...)
{
  static _Thread_local char buf[DAMACY_NVTX_FMT_BUF];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  damacy_nvtx_range_push(buf);
}

void
damacy_nvtx_range_pop(void)
{
  nvtxDomainRangePop(get_domain());
}

void
damacy_nvtx_mark(const char* name)
{
  if (!name)
    name = "(null)";
  nvtxEventAttributes_t ev = { 0 };
  ev.version = NVTX_VERSION;
  ev.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  ev.messageType = NVTX_MESSAGE_TYPE_ASCII;
  ev.message.ascii = name;
  nvtxDomainMarkEx(get_domain(), &ev);
}

void
damacy_nvtx_stream_name(CUstream s, const char* name)
{
  if (!s || !name)
    return;
  nvtxNameCuStreamA(s, name);
}

#endif // DAMACY_NVTX_ENABLED
