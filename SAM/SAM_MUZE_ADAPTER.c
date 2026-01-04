#include "../utils/NN/MUZE/all.h"
#include "SAM.h"
#include <stdlib.h>

static void sam_encode(void *brain, float *obs, size_t obs_dim,
                       long double **latent_seq, size_t *seq_len) {
  (void)brain;

  long double *input = malloc(sizeof(long double) * obs_dim);
  for (size_t i = 0; i < obs_dim; i++)
    input[i] = obs[i];

  long double **seq = malloc(sizeof(long double *));
  seq[0] = input;

  latent_seq = seq;
  *seq_len = 1;
}

static void sam_policy(void *brain, long double **latent_seq, size_t seq_len,
                       float *action_probs, size_t action_count) {
  SAM_t *sam = (SAM_t *)brain;

  long double *out = SAM_forward(sam, latent_seq, seq_len);

  for (size_t i = 0; i < action_count; i++)
    action_probs[i] = (float)out[i];

  free(out);
}

static void sam_learn(void *brain, float reward, int terminal) {
  SAM_t *sam = (SAM_t *)brain;
  SAM_update_context(sam, reward);

  if (terminal) {
    SAM_generalize(sam);
    SAM_transfuse(sam);
  }
}

MuCortex *SAM_as_MUZE(SAM_t *sam) {
  MuCortex *c = malloc(sizeof(MuCortex));
  c->brain = sam;
  c->encode = sam_encode;
  c->policy = sam_policy;
  c->learn = sam_learn;
  return c;
}
