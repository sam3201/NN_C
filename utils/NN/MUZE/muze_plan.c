#include "muze_plan.h"
#include "mcts.h"
#include <math.h>
#include <stdlib.h>

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
    if (x < 1e-20f)
      x = 1e-20f;
    p[i] = powf(x, invT);
  }
  renorm_probs(p, n);
}

static float rand01(MCTSRng *rng) {
  if (rng && rng->rand01)
    return rng->rand01(rng->ctx);
  return (float)rand() / (float)RAND_MAX;
}

static size_t rand_uniform_index(size_t n, MCTSRng *rng) {
  if (n == 0)
    return 0;
  float r = rand01(rng); // [0,1)
  size_t idx = (size_t)(r * (float)n);
  if (idx >= n)
    idx = n - 1;
  return idx;
}

static size_t sample_from_probs(const float *p, size_t n, MCTSRng *rng) {
  if (!p || n == 0)
    return 0;

  float r = rand01(rng);
  float c = 0.0f;

  for (size_t i = 0; i + 1 < n; i++) {
    c += p[i];
    if (r <= c)
      return i;
  }
  return n - 1;
}
/* ========================= */

int muze_plan(MuCortex *cortex, float *obs, size_t obs_dim, size_t action_count,
              MCTSRng *rng) {
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
      MCTSResult r = mcts_run_latent(cortex->mcts_model, latent,
                                     &cortex->mcts_params, rng);

      free(latent);

      if (cortex->free_latent_seq)
        cortex->free_latent_seq(cortex->brain, latent_seq, seq_len);

      int a = r.chosen_action;
      mcts_result_free(&r);
      return a;
    }
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

  renorm_probs(action_probs, action_count);

  float T = cortex->policy_temperature;
  if (!(T > 0.0f))
    T = 1.0f;
  apply_temperature_to_probs(action_probs, action_count, T);

  float eps = cortex->policy_epsilon;
  if (!(eps >= 0.0f))
    eps = 0.0f;
  if (eps > 1.0f)
    eps = 1.0f;

  size_t chosen;
  float u = (rng && rng->rand01) ? rng->rand01(rng->ctx)
                                 : (float)rand() / (float)RAND_MAX;
  if (u < eps) {
    chosen = rand_uniform_index(action_count, rng);
  } else {
    chosen = sample_from_probs(action_probs, action_count, rng);
  }

  if (cortex->free_latent_seq)
    cortex->free_latent_seq(cortex->brain, latent_seq, seq_len);

  free(action_probs);
  return (int)chosen;
}
