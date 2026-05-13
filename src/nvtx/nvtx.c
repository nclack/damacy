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

#define DAMACY_NVTX_FMT_BUF 128

// Lazily-created so we don't drag the NVTX init into damacy_create's
// hot path. nvtxDomainCreateA is thread-safe and idempotent enough at
// this scale (worst case: one extra unused domain handle on a race).
static nvtxDomainHandle_t s_domain = NULL;

static nvtxDomainHandle_t
get_domain(void)
{
  if (!s_domain)
    s_domain = nvtxDomainCreateA("damacy");
  return s_domain;
}

void
damacy_nvtx_init(void)
{
  (void)get_domain();
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
