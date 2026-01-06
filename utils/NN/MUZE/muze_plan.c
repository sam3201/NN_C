#include "muze_plan.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/*
  Minimal MUZE planner:
  - encodes obs -> latent sequence (owned by cortex)
  - queries policy
  - chooses action (argmax for now)
*/

int muze_plan(MuCortex *cortex, float *obs, size_t obs_dim,
              size_t action_count) {
  if (!cortex || !cortex->encode || !cortex->policy || !obs || obs_dim == 0 ||
      action_count == 0) {
    return 0;
  }

  long double **latent_seq = NULL;
  size_t seq_len = 0;

  cortex->encode(cortex->brain, obs, obs_dim, &latent_seq, &seq_len);

  float *action_probs = (float *)calloc(action_count, sizeof(float));
  if (!action_probs) {
    if (cortex->free_latent_seq)
      cortex->free_latent_seq(cortex->brain, latent_seq, seq_len);
    return 0;
  }

  cortex->policy(cortex->brain, latent_seq, seq_len, action_probs,
                 action_count);

  // Argmax
  int best = 0;
  float best_v = action_probs[0];
  for (size_t i = 1; i < action_count; i++) {
    if (action_probs[i] > best_v) {
      best_v = action_probs[i];
      best = (int)i;
    }
  }

  // Cleanup: latent_seq is cortex-owned.
  if (cortex->free_latent_seq)
    cortex->free_latent_seq(cortex->brain, latent_seq, seq_len);

  free(action_probs);
  return best;
}
