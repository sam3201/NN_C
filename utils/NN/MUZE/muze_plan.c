#include "muze_plan.h"
#include <stdlib.h>

/*
  Minimal MUZE planner:
  - encode obs -> latent sequence (owned/managed by cortex)
  - policy(latent_seq) -> action_probs (caller-owned)
  - choose action via argmax
*/

static int argmaxf(const float *x, size_t n) {
  if (!x || n == 0)
    return 0;

  size_t best = 0;
  float bestv = x[0];
  for (size_t i = 1; i < n; i++) {
    if (x[i] > bestv) {
      bestv = x[i];
      best = i;
    }
  }
  return (int)best;
}

int muze_plan(MuCortex *cortex, float *obs, size_t obs_dim,
              size_t action_count) {
  if (!cortex || !cortex->encode || !cortex->policy || !obs)
    return 0;
  if (obs_dim == 0 || action_count == 0)
    return 0;

  long double **latent_seq = NULL;
  size_t seq_len = 0;

  cortex->encode(cortex->brain, obs, obs_dim, &latent_seq, &seq_len);

  float *probs = (float *)calloc(action_count, sizeof(float));
  if (!probs) {
    if (cortex->free_latent_seq)
      cortex->free_latent_seq(cortex->brain, latent_seq, seq_len);
    return 0;
  }

  cortex->policy(cortex->brain, latent_seq, seq_len, probs, action_count);

  int action = argmaxf(probs, action_count);

  if (cortex->free_latent_seq)
    cortex->free_latent_seq(cortex->brain, latent_seq, seq_len);

  free(probs);
  return action;
}
