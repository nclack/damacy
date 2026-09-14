#pragma once

#include "numa/numa.h"
#include <cuda.h>

void
numa_init(enum damacy_numa_strategy strategy,
          int override_node,
          CUdevice device,
          struct numa_resolved* out);
