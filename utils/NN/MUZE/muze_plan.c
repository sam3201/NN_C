#include "muze_plan.h"
#include "mcts.h"
#include <stdlib.h>
#include <string.h>

/* Argmax helper */
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

/* Convert latent_seq (seq_len x obs_dim long doubles) -> float
   latent[latent_dim] For now: use the LAST observation in the sequence, and
   copy as floats. If your MuModel expects a different encoding, swap this
   function.
*/
static void latent_seq_to_float_latent(long double **latent_seq, size_t seq_len,
                                       size_t obs_dim, float *out_latent,
                                       size_t latent_dim) {
  /* zero init */
  for (size_t i = 0; i < latent_dim; i++)
    out_latent[i] = 0.0f;

  if (!latent_seq || seq_len == 0 || !latent_seq[seq_len - 1])
    return;

  long double *last = latent_seq[seq_len - 1];

  size_t n = obs_dim < latent_dim ? obs_dim : latent_dim;
  for (size_t i = 0; i < n; i++)
    out_latent[i] = (float)last[i];
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

  /* ------------------------------------------
     Option A: Use MCTS (if enabled + model set)
     ------------------------------------------ */
  if (cortex->use_mcts && cortex->mcts_model) {
    const int L = cortex->mcts_model->cfg.latent_dim;

    float *latent = (float *)malloc(sizeof(float) * (size_t)L);
    if (latent) {
      latent_seq_to_float_latent(latent_seq, seq_len, obs_dim, latent,
                                 (size_t)L);

      /* Run MCTS directly on latent */
      MCTSResult r =
          mcts_run_latent(cortex->mcts_model, latent, &cortex->mcts_params);
      free(latent);

      if (cortex->free_latent_seq)
        cortex->free_latent_seq(cortex->brain, latent_seq, seq_len);

      int chosen = r.chosen_action;
      mcts_result_free(&r);
      return chosen;
    }
    /* If latent alloc failed, fall through to policy */
  }

  /* ------------------------------------------
     Option B: Policy -> argmax
     ------------------------------------------ */
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
