#ifndef MUZE_CORTEX_H
#define MUZE_CORTEX_H

#include "../../../SAM/SAM.h"
#include <stdio.h>

typedef struct {
  void *brain;

  void (*encode)(void *brain, float *obs, size_t obs_dim,
                 long double ***latent_seq, size_t *seq_len);

  void (*policy)(void *brain, long double **latent_seq, size_t seq_len,
                 float *action_probs, size_t action_count);

  void (*learn)(void *brain, float reward, int terminal);
  void free_latent_seq(void *brain, long double **latent_seq, size_t seq_len);

} MuCortex;

#endif

MuCortex *SAM_as_MUZE(SAM_t *sam);
