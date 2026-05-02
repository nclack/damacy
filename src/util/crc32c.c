#include "util/crc32c.h"

#include "platform/platform.h"

static uint32_t crc32c_table[256];
static platform_once crc32c_once = PLATFORM_ONCE_INIT;

static void
crc32c_init(void)
{
  for (int i = 0; i < 256; ++i) {
    uint32_t crc = (uint32_t)i;
    for (int j = 0; j < 8; ++j)
      crc = (crc >> 1) ^ (0x82F63B78u & (0u - (crc & 1)));
    crc32c_table[i] = crc;
  }
}

uint32_t
crc32c(const void* data, size_t len)
{
  platform_call_once(&crc32c_once, crc32c_init);
  uint32_t crc = 0xFFFFFFFFu;
  const uint8_t* p = (const uint8_t*)data;
  for (size_t i = 0; i < len; ++i)
    crc = crc32c_table[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
  return crc ^ 0xFFFFFFFFu;
}
