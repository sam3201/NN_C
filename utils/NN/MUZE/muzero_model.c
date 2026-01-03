#include "muzero_model.h"
#include "runtime.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------
   Create model
   ------------------------ */
MuModel *mu_model_create(const MuConfig *cfg) {
  MuModel *m = (MuModel *)malloc(sizeof(MuModel));
  m->cfg = *cfg;

  int obs = cfg->obs_dim;
  int lat = cfg->latent_dim;
  int act = cfg->action_count;

  /* allocate fake/placeholder weights */
  m->repr_W_count = obs * lat;
  m->dyn_W_count = (lat + 1) * lat;  // +1 for action embedding
  m->pred_W_count = lat * (act + 1); // policy + value head

  m->repr_W = (float *)malloc(sizeof(float) * m->repr_W_count);
  m->dyn_W = (float *)malloc(sizeof(float) * m->dyn_W_count);
  m->pred_W = (float *)malloc(sizeof(float) * m->pred_W_count);

  /* simple initialization */
  for (int i = 0; i < m->repr_W_count; i++)
    m->repr_W[i] = 0.01f;
  for (int i = 0; i < m->dyn_W_count; i++)
    m->dyn_W[i] = 0.01f;
  for (int i = 0; i < m->pred_W_count; i++)
    m->pred_W[i] = 0.01f;

  m->runtime = mu_runtime_create(m, 4096, 0.95f);

  return m;
}

/* ------------------------
   Free model
   ------------------------ */
void mu_model_free(MuModel *m) {
  if (!m)
    return;
  free(m->repr_W);
  free(m->dyn_W);
  free(m->pred_W);
  free(m);
  mu_runtime_free(m->runtime);
}

/* ------------------------
   Representation function
   obs → latent
   (Dummy linear layer)
   ------------------------ */
void mu_model_repr(MuModel *m, const float *obs, float *latent_out) {
  int O = m->cfg.obs_dim;
  int L = m->cfg.latent_dim;

  for (int i = 0; i < L; i++) {
    float sum = 0.f;
    for (int j = 0; j < O; j++) {
      sum += obs[j] * m->repr_W[i * O + j];
    }
    latent_out[i] = tanhf(sum);
  }
}

/* ------------------------
   Dynamics function
   latent + action → latent' + reward
   ------------------------ */
void mu_model_dynamics(MuModel *m, const float *latent_in, int action,
                       float *latent_out, float *reward_out) {
  int L = m->cfg.latent_dim;

  /* simple deterministic dynamics */
  for (int i = 0; i < L; i++) {
    float sum = 0.f;
    for (int j = 0; j < L; j++) {
      sum += latent_in[j] * m->dyn_W[i * L + j];
    }
    sum += 0.1f * action;
    latent_out[i] = tanhf(sum);
  }

  *reward_out = 0.01f * action; // placeholder
}

/* ------------------------
   Prediction function
   latent → (policy_logits, value)
   ------------------------ */
void mu_model_predict(MuModel *m, const float *latent_in,
                      float *policy_logits_out, float *value_out) {
  int L = m->cfg.latent_dim;
  int A = m->cfg.action_count;

  /* policy */
  for (int a = 0; a < A; a++) {
    float sum = 0.f;
    for (int j = 0; j < L; j++) {
      sum += latent_in[j] * m->pred_W[a * L + j];
    }
    policy_logits_out[a] = sum;
  }

  /* value head */
  float sum = 0.f;
  for (int j = 0; j < L; j++) {
    sum += latent_in[j] * m->pred_W[(A * L) + j];
  }
  *value_out = tanhf(sum);
}

#include "runtime.h"

void mu_model_step(MuModel *m, const float *obs, int action, float reward) {
  mu_runtime_step((MuRuntime *)m->runtime, m, obs, action, reward);
}

void mu_model_end_episode(MuModel *m, float terminal_reward) {
  mu_runtime_end_episode((MuRuntime *)m->runtime, m, terminal_reward);
}

void mu_model_reset_episode(MuModel *m) {
  mu_runtime_reset_episode((MuRuntime *)m->runtime);
}

void mu_model_train(MuModel *m) {
  mu_runtime_train((MuRuntime *)m->runtime, m);
}
