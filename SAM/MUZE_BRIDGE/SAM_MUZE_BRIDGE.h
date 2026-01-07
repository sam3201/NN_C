#ifndef SAM_MUZE_BRIDGE_H
#define SAM_MUZE_BRIDGE_H

#pragma once
#include "../utils/NN/MUZE/muze_cortex.h"
#include "SAM.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Creates a MUZE cortex wrapper around SAM. Caller owns returned MuCortex*. */
MuCortex *SAM_as_MUZE(SAM_t *sam);

/* Frees cortex created by SAM_as_MUZE(). */
void SAM_MUZE_destroy(MuCortex *cortex);

#ifdef __cplusplus
}
#endif

#endif // SAM_MUZE_BRIDGE_H
