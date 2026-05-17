#include "damacy.h"

const char*
damacy_status_str(enum damacy_status s)
{
  switch (s) {
    case DAMACY_OK:
      return "ok";
    case DAMACY_AGAIN:
      return "again";
    case DAMACY_INVAL:
      return "invalid argument";
    case DAMACY_NOTFOUND:
      return "not found";
    case DAMACY_DTYPE:
      return "dtype mismatch";
    case DAMACY_RANK:
      return "rank mismatch";
    case DAMACY_IO:
      return "io error";
    case DAMACY_DECODE:
      return "decode error";
    case DAMACY_CUDA:
      return "cuda error";
    case DAMACY_OOM:
      return "out of memory";
    case DAMACY_BUDGET:
      return "configured budget too small";
    case DAMACY_SHUTDOWN:
      return "shutdown";
  }
  return "unknown";
}
