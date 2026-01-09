#include "runtime.h"
#include "trainer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static TrainerConfig trainer_default_cfg(void) {
  TrainerConfig tc = {
      .batch_size = 32,
      .train_steps = 200,
      .min_replay_size = TRAIN_WARMUP,
      .lr = 0.05f,
  };
  return tc;
}

MuRuntime *mu_runtime_create(MuModel *model, float gamma) {
  MuRuntime *rt = calloc(1, sizeof(MuRuntime));
  if (!rt)
    return NULL;

  if (!model) {
    free(rt);
    return NULL;
  }

  rt->rb = rb_create(TRAIN_WINDOW, model->cfg.obs_dim, model->cfg.action_count);
  rt->last_obs = malloc(sizeof(float) * (size_t)model->cfg.obs_dim);
  rt->last_pi = malloc(sizeof(float) * (size_t)model->cfg.action_count);
  if (rt->last_pi)
    memset(rt->last_pi, 0, sizeof(float) * (size_t)model->cfg.action_count);

  rt->has_last = 0;
  rt->gamma = gamma;
  rt->total_steps = 0;

  rt->cfg = trainer_default_cfg();
  rt->has_cfg = 0;

  if (!rt->rb || !rt->last_obs || !rt->last_pi) {
    if (rt->rb)
      rb_free(rt->rb);
    free(rt->last_obs);
    free(rt->last_pi);
    free(rt);
    return NULL;
  }

  return rt;
}

void mu_runtime_set_trainer_config(MuRuntime *rt, const TrainerConfig *cfg) {
  if (!rt)
    return;
  if (cfg) {
    rt->cfg = *cfg;
    rt->has_cfg = 1;
  } else {
    rt->cfg = trainer_default_cfg();
    rt->has_cfg = 0;
  }
}

TrainerConfig mu_runtime_get_trainer_config(const MuRuntime *rt) {
  if (!rt)
    return trainer_default_cfg();
  return rt->has_cfg ? rt->cfg : trainer_default_cfg();
}

void mu_runtime_free(MuRuntime *rt) {
  if (!rt)
    return;
  rb_free(rt->rb);
  free(rt->last_obs);
  free(rt->last_pi);
  free(rt);
}

// runtime.c
void mu_runtime_step(MuRuntime *rt, MuModel *model, const float *obs,
                     int action, float reward) {
  if (!rt || !model || !obs)
    return;

  const int O = model->cfg.obs_dim;
  const int A = model->cfg.action_count;

  // Build a one-hot pi for THIS decision (action taken at obs)
  float *pi = (float *)malloc(sizeof(float) * (size_t)A);
  if (!pi)
    return;

  for (int i = 0; i < A; i++)
    pi[i] = 0.0f;
  if (action >= 0 && action < A)
    pi[action] = 1.0f;

  // Use the correct caching/pushing logic
  mu_runtime_step_with_pi(rt, model, obs, pi, action, reward);

  free(pi);
}

void mu_runtime_step_with_pi(MuRuntime *rt, MuModel *model, const float *obs,
                             const float *pi, int action, float reward) {
  if (!rt || !model || !obs || !pi)
    return;

  rt->total_steps++;

  const int O = model->cfg.obs_dim;
  const int A = model->cfg.action_count;

  if (!rt->has_last) {
    memcpy(rt->last_obs, obs, sizeof(float) * (size_t)O);
    memcpy(rt->last_pi, pi, sizeof(float) * (size_t)A);
    rt->last_action = action;
    rt->has_last = 1;
    return;
  }

  // We just received reward from last_action taken at last_obs,
  // and obs is the resulting next_obs.
  float z = reward; // keep your current 1-step target for now

  rb_push_full(rt->rb, rt->last_obs, rt->last_pi, z, rt->last_action, reward,
               obs,
               /*done*/ 0);

  // cache current decision for next step
  memcpy(rt->last_obs, obs, sizeof(float) * (size_t)O);
  memcpy(rt->last_pi, pi, sizeof(float) * (size_t)A);
  rt->last_action = action;
}

void mu_runtime_end_episode(MuRuntime *rt, MuModel *model,
                            float terminal_reward) {
  if (!rt || !model)
    return;
  if (!rt->has_last)
    return;

  float z = terminal_reward;

  rb_push_full(rt->rb, rt->last_obs, rt->last_pi, z, rt->last_action,
               terminal_reward,
               rt->last_obs, // no next obs; reuse
               /*done*/ 1);

  rt->has_last = 0;
}

void mu_runtime_reset_episode(MuRuntime *rt) { rt->has_last = 0; }

void mu_runtime_train(MuRuntime *rt, MuModel *model, const TrainerConfig *cfg) {
  if (!rt || !model)
    return;

  // Priority:
  // 1) per-call cfg if provided
  // 2) runtime-stored cfg if user set it
  // 3) defaults
  TrainerConfig tc;
  if (cfg)
    tc = *cfg;
  else if (rt->has_cfg)
    tc = rt->cfg;
  else
    tc = trainer_default_cfg();

  // policy/value pass (obs,pi,z)
  trainer_train_from_replay(model, rt->rb, &tc);

  // dynamics/reward pass (obs,a,r,next_obs,done)
  trainer_train_dynamics(model, rt->rb, &tc);
}

static void normalize_probs(float *p, size_t n) {
  float sum = 0.0f;
  for (size_t i = 0; i < n; i++) {
    if (p[i] < 0.0f)
      p[i] = 0.0f;
    sum += p[i];
  }
  if (sum <= 1e-12f) {
    float u = (n ? 1.0f / (float)n : 0.0f);
    for (size_t i = 0; i < n; i++)
      p[i] = u;
    return;
  }
  float inv = 1.0f / sum;
  for (size_t i = 0; i < n; i++)
    p[i] *= inv;
}

static void apply_temperature(float *p, size_t n, float temp) {
  // temp <= 0 => act like argmax (very sharp). We'll approximate by hard
  // one-hot.
  if (n == 0)
    return;

  if (temp <= 0.0f) {
    size_t best = 0;
    float bestv = p[0];
    for (size_t i = 1; i < n; i++) {
      if (p[i] > bestv) {
        bestv = p[i];
        best = i;
      }
    }
    for (size_t i = 0; i < n; i++)
      p[i] = (i == best) ? 1.0f : 0.0f;
    return;
  }

  // Standard: p_i <- p_i^(1/temp)
  float invT = 1.0f / temp;
  for (size_t i = 0; i < n; i++) {
    float x = p[i];
    // avoid powf(0, something)
    p[i] = (x <= 0.0f) ? 0.0f : powf(x, invT);
  }
  normalize_probs(p, n);
}

static void apply_epsilon_mix(float *p, size_t n, float eps) {
  if (n == 0)
    return;
  if (eps <= 0.0f)
    return;
  if (eps > 1.0f)
    eps = 1.0f;

  float u = 1.0f / (float)n;
  for (size_t i = 0; i < n; i++) {
    p[i] = (1.0f - eps) * p[i] + eps * u;
  }
  normalize_probs(p, n);
}

static int sample_from_probs(const float *p, size_t n, MCTSRng *rng) {
  if (n == 0)
    return -1;

  float r = 0.0f;
  if (rng && rng->rand01)
    r = rng->rand01(rng->ctx);
  else
    r = (float)rand() / (float)RAND_MAX;

  float c = 0.0f;
  for (size_t i = 0; i < n; i++) {
    c += p[i];
    if (r <= c)
      return (int)i;
  }
  return (int)(n - 1);
}

int muze_select_action(MuCortex *cortex, const float *obs, size_t obs_dim,
                       float *out_pi, size_t action_count, MCTSRng *rng) {
  if (!cortex || !obs || !out_pi || action_count == 0)
    return -1;

  // ---- Case 1: MCTS ----
  if (cortex->use_mcts) {
    if (!cortex->mcts_model)
      return -1;

    // NOTE: match your actual signature:
    // In your jump.c you used: mcts_run(model, obs, &params, &rng)
    MCTSResult mr =
        mcts_run(cortex->mcts_model, obs, &cortex->mcts_params, rng);

    // Copy pi out
    +size_t n = action_count;
    if ((size_t)mr.action_count < n)
      n = (size_t)mr.action_count;
    memcpy(out_pi, mr.pi, sizeof(float) * n);
    // if caller provided larger buffer, pad remainder uniformly
    for (size_t i = n; i < action_count; i++)
      out_pi[i] = 0.0f;

    normalize_probs(out_pi, n);
    apply_temperature(out_pi, n, cortex->policy_temperature);
    apply_epsilon_mix(out_pi, n, cortex->policy_epsilon);

    int a = sample_from_probs(out_pi, n, rng);

    mcts_result_free(&mr);
    return a;
  }

  // ---- Case 2: Direct policy (SAM bridge, etc.) ----
  if (!cortex->encode || !cortex->policy)
    return -1;

  long double **latent_seq = NULL;
  size_t seq_len = 0;

  // encode may want non-const float*
  cortex->encode(cortex->brain, (float *)obs, obs_dim, &latent_seq, &seq_len);

  if (!latent_seq || seq_len == 0) {
    if (cortex->free_latent_seq)
      cortex->free_latent_seq(cortex->brain, latent_seq, seq_len);
    // fallback to uniform
    float u = 1.0f / (float)action_count;
    for (size_t i = 0; i < action_count; i++)
      out_pi[i] = u;
    return sample_from_probs(out_pi, action_count, rng);
  }

  cortex->policy(cortex->brain, latent_seq, seq_len, out_pi, action_count);

  if (cortex->free_latent_seq)
    cortex->free_latent_seq(cortex->brain, latent_seq, seq_len);

  normalize_probs(out_pi, action_count);
  apply_temperature(out_pi, action_count, cortex->policy_temperature);
  apply_epsilon_mix(out_pi, action_count, cortex->policy_epsilon);

  return sample_from_probs(out_pi, action_count, rng);
}
