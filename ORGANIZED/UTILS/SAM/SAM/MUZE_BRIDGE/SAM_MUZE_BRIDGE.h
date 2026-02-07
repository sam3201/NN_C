#pragma once
#include "../../SAM/SAM.h"
#include "../../utils/NN/MUZE/muze_cortex.h" // or wherever MuCortex is declared

#ifdef __cplusplus
extern "C" {
#endif

// Create a MuCortex wrapper around a SAM brain.
// Returned MuCortex owns no SAM memory unless you decide it does.
MuCortex *SAM_as_MUZE(SAM_t *sam);

// Free only the cortex wrapper (not the SAM brain), unless you choose
// otherwise.
void SAM_as_MUZE_destroy(MuCortex *cortex);

#ifdef __cplusplus
}
#endif
