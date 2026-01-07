#include "muze_plan.h"
#include "mcts.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---------- helpers ---------- */

static void renorm_probs(float *p, size_t n) {
  float sum = 0.0f;
  for (size_t i = 0; i < n; i++) {
    if (!isfinite(p[i]) || p[i] < 0.0f)
      p[i] = 0.0f;
    sum += p[i];
  }
  if (sum > 0.0f) {
    float inv = 1.0f / sum;
    for (size_t i = 0; i < n; i++)
      p[i] *= inv;
  } else {
    float u = (n > 0) ? (1.0f / (float)n) : 0.0f;
    for (size_t i = 0; i < n; i++)
      p[i] = u;
  }
}

static void apply_temperature_to_probs(float *p, size_t n, float temperature) {
  /* We assume p is already a probability distribution.
     Apply: p_i <- p_i^(1/T) then renormalize.
     - T=1: no change
     - T<1: sharper
     - T>1: flatter
  */
  if (n == 0)
    return;

  if (!(temperature > 0.0f))
    temperature = 1e-6f;
  if (fabsf(temperature - 1.0f) < 1e-6f) {
    renorm_probs(p, n);
    return;
  }

  float invT = 1.0f / temperature;
  for (size_t i = 0; i < n; i++) {
    float x = p[i];
    if (!isfinite(x) || x < 0.0f)
      x = 0.0f;
    /* clamp tiny values so powf behaves nicely */
    if (x < 1e-20f)
      x = 1e-20f;
    p[i] = powf(x, invT);
  }
  renorm_probs(p, n);
}

static size_t sample_from_probs(const float *p, size_t n) {
  if (!p || n == 0)
    return 0;
  float r = (float)rand() / (float)RAND_MAX;
  float c = 0.0f;
  for (size_t i = 0; i < n; i++) {
    c += p[i];
    if (r <= c)
      return i;
  }
  return n - 1;
}

static size_t rand_uniform_index(size_t n) {
  if (n == 0)
    return 0;
  return (size_t)(rand() % (int)n);
}

/* ========================= */

int muze_plan(MuCortex *cortex, float *obs, size_t obs_dim,
              size_t action_count) {
  if (!cortex || !cortex->encode || !obs)
    return 0;
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

  /* --- Option: policy sampling (non-MCTS) --- */
  if (!cortex->policy) {
    if (cortex->free_latent_seq)
      cortex->free_latent_seq(cortex->brain, latent_seq, seq_len);
    return 0;
  }

  float *action_probs = (float *)calloc(action_count, sizeof(float));
  if (!action_probs) {
    if (cortex->free_latent_seq)
      cortex->free_latent_seq(cortex->brain, latent_seq, seq_len);
    return 0;
  }

  cortex->policy(cortex->brain, latent_seq, seq_len, action_probs,
                 action_count);

  /* Safety: ensure normalized before transforms */
  renorm_probs(action_probs, action_count);

  /* Optional temperature (defaults to 1.0 if unset/invalid) */
  float T = cortex->policy_temperature;
  if (!(T > 0.0f))
    T = 1.0f;
  apply_temperature_to_probs(action_probs, action_count, T);

  /* Optional epsilon-greedy (defaults to 0.0 if unset/invalid) */
  float eps = cortex->policy_epsilon;
  if (!(eps >= 0.0f))
    eps = 0.0f;
  if (eps > 1.0f)
    eps = 1.0f;

  size_t chosen;
  float u = (float)rand() / (float)RAND_MAX;
  if (u < eps) {
    chosen = rand_uniform_index(action_count);
  } else {
    chosen = sample_from_probs(action_probs, action_count);
  }

  if (cortex->free_latent_seq)
    cortex->free_latent_seq(cortex->brain, latent_seq, seq_len);

  free(action_probs);
  return (int)chosen;
}

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
int muze_plan(MuCortex *cortex, float *obs, size_t obs_dim,
              size_t action_count) {
  if (!cortex || !cortex->encode || !obs)
    return 0;
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
  if (!cortex->policy) {
    if (cortex->free_latent_seq)
      cortex->free_latent_seq(cortex->brain, latent_seq, seq_len);
    return 0;
  }

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
