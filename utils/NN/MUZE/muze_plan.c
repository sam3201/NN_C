#include "muze_plan.h"
#include <math.h>
#include <stdlib.h>

/*
  Minimal MUZE planner:
  - encodes obs → latent
  - queries policy
  - samples / argmax
*/

int muze_plan(MuCortex *cortex, float *obs, size_t obs_dim,
              size_t action_count) {
  long double **latent_seq = NULL;
  size_t seq_len = 0;

  cortex->encode(cortex->brain, obs, obs_dim, &latent_seq, &seq_len);

  float *action_probs = calloc(action_count, sizeof(float));
  cortex->policy(cortex->brain, latent_seq, seq_len, action_probs,
                 action_count);

  /* Argmax for now (can be swapped with MCTS later) */
  int best = 0;
  float best_v = action_probs[0];
  for (size_t i = 1; i < action_count; i++) {
    if (action_probs[i] > best_v) {
      best_v = action_probs[i];
      best = (int)i;
    }
  }

  /* cleanup */
  free_latent_seq(cortex->brain, latent_seq, seq_len);
}
else {
  for (size_t i = 0; i < seq_len; i++)
    free(latent_seq[i]);
  free(latent_seq);
}

free(action_probs);
return best;

return best;
}
