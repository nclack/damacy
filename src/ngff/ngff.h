#pragma once

#include "damacy_spatial.h"
#include "dtype/dtype.h"
#include "util/json.h"

struct damacy_ngff_image
{
  struct damacy_ngff_info info;
  struct damacy_ngff_level* levels;
  enum dtype dtype;
};

enum damacy_status
ngff_parse_group(struct cslice src,
                 const char* uri,
                 uint32_t multiscale_index,
                 uint32_t max_levels,
                 struct damacy_ngff_image** out);
enum damacy_status
ngff_parse_array(struct cslice src,
                 struct damacy_ngff_image* image,
                 uint32_t level);
