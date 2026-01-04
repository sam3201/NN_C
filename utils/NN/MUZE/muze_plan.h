#ifndef MUZE_PLAN_H
#define MUZE_PLAN_H

#include "muze_cortex.h"

int muze_plan(MuCortex *cortex, float *obs, size_t obs_dim,
              size_t action_count);

#endif
