#ifndef MUZE_CORTEX_H
#define MUZE_CORTEX_H

#include <stdio.h>

typedef struct {
  void *brain; // opaque (SAM_t*)

  // perception → latent
  void (*encode)(void *brain, float *obs, size_t obs_dim,
                 long double **latent_seq, size_t *seq_len);

  // latent → action logits
  void (*policy)(void *brain, long double **latent_seq, size_t seq_len,
                 float *action_probs, size_t action_count);

  // learning signal
  void (*learn)(void *brain, float reward, int terminal);
} MuCortex;

#endif

MuCortex *SAM_as_MUZE(SAM_t *sam) {
  MuCortex *c = malloc(sizeof(MuCortex));
  c->brain = sam;
  c->encode = sam_encode;
  c->policy = sam_policy;
  c->learn = sam_learn;
  return c;
}
