#include "../utils/NN/MUZE/all.h"
#include "SAM.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* softmax for action logits -> probs */
static void softmaxf_inplace(float *x, size_t n) {
  if (n == 0)
    return;
  float maxv = x[0];
  for (size_t i = 1; i < n; i++)
    if (x[i] > maxv)
      maxv = x[i];

  float sum = 0.0f;
  for (size_t i = 0; i < n; i++) {
    x[i] = expf(x[i] - maxv);
    sum += x[i];
  }
  if (sum <= 0.0f) {
    float u = 1.0f / (float)n;
    for (size_t i = 0; i < n; i++)
      x[i] = u;
    return;
  }
  for (size_t i = 0; i < n; i++)
    x[i] /= sum;
}

static void sam_encode(void *brain, float *obs, size_t obs_dim,
                       long double ***latent_seq_out, size_t *seq_len_out) {
  (void)brain;

  if (!latent_seq_out || !seq_len_out || !obs || obs_dim == 0)
    return;

  long double *input = (long double *)malloc(sizeof(long double) * obs_dim);
  if (!input)
    return;

  for (size_t i = 0; i < obs_dim; i++)
    input[i] = (long double)obs[i];

  long double **seq = (long double **)malloc(sizeof(long double *));
  if (!seq) {
    free(input);
    return;
  }

  seq[0] = input;

  *latent_seq_out = seq;
  *seq_len_out = 1;
}

static void sam_policy(void *brain, long double **latent_seq, size_t seq_len,
                       float *action_probs, size_t action_count) {
  SAM_t *sam = (SAM_t *)brain;
  if (!sam || !latent_seq || seq_len == 0 || !action_probs || action_count == 0)
    return;

  long double *out = SAM_forward(sam, latent_seq, seq_len);
  if (!out)
    return;

  /* copy logits */
  for (size_t i = 0; i < action_count; i++)
    action_probs[i] = (float)out[i];

  /* turn logits into a proper probability distribution */
  softmaxf_inplace(action_probs, action_count);

  free(out);
}

static void sam_learn(void *brain, float reward, int terminal) {
  SAM_t *sam = (SAM_t *)brain;
  if (!sam)
    return;

  SAM_update_context(sam, (long double)reward);

  if (terminal) {
    SAM_generalize(sam);
    SAM_transfuse(sam);
  }
}

MuCortex *SAM_as_MUZE(SAM_t *sam) {
  MuCortex *c = (MuCortex *)malloc(sizeof(MuCortex));
  if (!c)
    return NULL;
  c->brain = sam;
  c->encode = sam_encode;
  c->policy = sam_policy;
  c->learn = sam_learn;
  return c;
}
