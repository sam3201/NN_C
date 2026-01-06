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

/* Convert cortex latent_seq (seq_len x obs_dim long doubles) to float
   latent[L]. Right now: take the LAST frame in the sequence and copy as floats.
   This is enough to let you plug MCTS in and visualize it end-to-end.
   Later, if you want "true MuZero latent", we'll wire model->repr to produce
   it.
*/
static void latent_seq_to_latent_float(long double **latent_seq, size_t seq_len,
                                       size_t obs_dim, float *latent_out,
                                       size_t latent_dim) {
  /* zero */
  for (size_t i = 0; i < latent_dim; i++)
    latent_out[i] = 0.0f;

  if (!latent_seq || seq_len == 0 || !latent_seq[seq_len - 1])
    return;

  long double *last = latent_seq[seq_len - 1];
  size_t n = obs_dim < latent_dim ? obs_dim : latent_dim;
  for (size_t i = 0; i < n; i++)
    latent_out[i] = (float)last[i];
}

int muze_plan(MuCortex *cortex, float *obs, size_t obs_dim,
              size_t action_count) {

  if (obs_dim == 0 || action_count == 0)
    return 0;

  long double **latent_seq = NULL;
  size_t seq_len = 0;

  cortex->encode(cortex->brain, obs, obs_dim, &latent_seq, &seq_len);

  /* --- Option: MCTS --- */
  if (cortex->use_mcts && cortex->mcts_model) {
    size_t L = (size_t)cortex->mcts_model->cfg.latent_dim;
    float *latent = (float *)malloc(sizeof(float) * L);

    if (latent) {
      mu_model_repr(cortex->mcts_model, obs, latent);
      MCTSResult r =
          mcts_run_latent(cortex->mcts_model, latent, &cortex->mcts_params);

      free(latent);

      if (cortex->free_latent_seq)
        cortex->free_latent_seq(cortex->brain, latent_seq, seq_len);

      int a = r.chosen_action;
      mcts_result_free(&r);
      return a;
    }
    /* if malloc failed, fall through to policy */
  }

  /* --- Option: policy argmax --- */
  float *action_probs = (float *)calloc(action_count, sizeof(float));
  if (!action_probs) {
    if (cortex->free_latent_seq)
      cortex->free_latent_seq(cortex->brain, latent_seq, seq_len);
    return 0;
  }

  cortex->policy(cortex->brain, latent_seq, seq_len, action_probs,
                 action_count);

  int best = argmaxf(action_probs, action_count);

  if (cortex->free_latent_seq)
    cortex->free_latent_seq(cortex->brain, latent_seq, seq_len);

  free(action_probs);
  return best;
}
