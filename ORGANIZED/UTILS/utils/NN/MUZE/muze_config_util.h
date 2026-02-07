#ifndef MUZE_CONFIG_UTIL_H
#define MUZE_CONFIG_UTIL_H

#include "muze_config.h"

#ifdef __cplusplus
extern "C" {
#endif

void muze_config_init_defaults(MuzeConfig *cfg, int obs_dim, int action_count);

#ifdef __cplusplus
}
#endif

#endif
