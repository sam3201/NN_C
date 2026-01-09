#include "muzero_model.h"
#include "replay_buffer.h"
#include "runtime.h"
#include "trainer.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------
   Helpers
   ------------------------ */
static void default_repr(MuModel *m, const float *obs, float *latent_out) {
  int O = m->cfg.obs_dim;
  int L = m->cfg.latent_dim;
  for (int i = 0; i < L; i++) {
    float sum = 0.f;
    for (int j = 0; j < O; j++)
      sum += obs[j] * m->repr_W[i * O + j];
    latent_out[i] = tanhf(sum);
  }
}

static void default_dynamics(MuModel *m, const float *latent_in, int action,
                             float *latent_out, float *reward_out) {
  int L = m->cfg.latent_dim;

  float a = (float)action / (float)(m->cfg.action_count - 1); // 0..1
  float a2 = a * 2.0f - 1.0f;                                 // -1..1

  for (int i = 0; i < L; i++) {
    float sum = 0.f;
    for (int j = 0; j < L; j++)
      sum += latent_in[j] * m->dyn_W[i * (L + 1) + j];
    sum += a2 * m->dyn_W[i * (L + 1) + L];
    latent_out[i] = tanhf(sum);
  }

  // reward head: r = dot(rew_W, latent_out) + b
  float r = m->rew_b;
  for (int i = 0; i < L; i++)
    r += m->rew_W[i] * latent_out[i];
  *reward_out = tanhf(r); // squash optional
}

static void default_predict(MuModel *m, const float *latent_in,
                            float *policy_logits_out, float *value_out) {
  int L = m->cfg.latent_dim;
  int A = m->cfg.action_count;

  for (int a = 0; a < A; a++) {
    float sum = 0.f;
    for (int j = 0; j < L; j++)
      sum += latent_in[j] * m->pred_W[a * L + j];
    policy_logits_out[a] = sum;
  }

  float sum = 0.f;
  for (int j = 0; j < L; j++)
    sum += latent_in[j] * m->pred_W[(A * L) + j];
  *value_out = tanhf(sum);
}

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
  m->dyn_W_count = (lat) * (lat + 1);

  m->pred_W_count = lat * (act + 1); // policy + value head

  m->repr_W = (float *)malloc(sizeof(float) * m->repr_W_count);
  m->dyn_W = malloc(sizeof(float) * m->dyn_W_count);
  m->pred_W = (float *)malloc(sizeof(float) * m->pred_W_count);

  m->rew_W_count = lat;
  m->rew_W = (float *)malloc(sizeof(float) * lat);
  m->rew_b = 0.0f;
  for (int i = 0; i < lat; i++)
    m->rew_W[i] = 0.01f;

  /* simple initialization */
  for (int i = 0; i < m->repr_W_count; i++)
    m->repr_W[i] = 0.01f;
  for (int i = 0; i < m->dyn_W_count; i++)
    m->dyn_W[i] = 0.01f;
  for (int i = 0; i < m->pred_W_count; i++)
    m->pred_W[i] = 0.01f;

  m->runtime = mu_runtime_create(m, 0.95f);

  m->repr = NULL;
  m->dynamics = NULL;
  m->predict = NULL;

  return m;
}

/* ------------------------
   Free model
   ------------------------ */
void mu_model_free(MuModel *m) {
  if (!m)
    return;
  if (m->runtime)
    mu_runtime_free((MuRuntime *)m->runtime);
  free(m->repr_W);
  free(m->dyn_W);
  free(m->pred_W);
  free(m->rew_W);
  free(m);
}

/* ------------------------
   Representation function
   obs → latent
   (Dummy linear layer)
   ------------------------ */
void mu_model_repr(MuModel *m, const float *obs, float *latent_out) {
  if (!m)
    return;
  if (m->repr) {
    m->repr(m, obs, latent_out);
    return;
  } else {
    default_repr(m, obs, latent_out);
  }
}

void mu_model_dynamics(MuModel *m, const float *latent_in, int action,
                       float *latent_out, float *reward_out) {
  if (!m)
    return;
  if (m->dynamics) {
    m->dynamics(m, latent_in, action, latent_out, reward_out);
    return;
  } else {
    default_dynamics(m, latent_in, action, latent_out, reward_out);
  }
}

void mu_model_predict(MuModel *m, const float *latent_in,
                      float *policy_logits_out, float *value_out) {
  if (!m)
    return;
  if (m->predict) {
    m->predict(m, latent_in, policy_logits_out, value_out);
    return;
  } else {
    default_predict(m, latent_in, policy_logits_out, value_out);
  }
}

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
  mu_runtime_train((MuRuntime *)m->runtime, m, NULL);
}

void mu_model_train_with_cfg(MuModel *m, const TrainerConfig *cfg) {
  mu_runtime_train((MuRuntime *)m->runtime, m, cfg);
}

int muzero_model_obs_dim(MuModel *m) { return m ? m->cfg.obs_dim : 0; }
int muzero_model_action_count(MuModel *m) {
  return m ? m->cfg.action_count : 0;
}

static void softmaxf(const float *logits, int A, float *out_probs) {
  float maxv = logits[0];
  for (int i = 1; i < A; i++)
    if (logits[i] > maxv)
      maxv = logits[i];

  float sum = 0.0f;
  for (int i = 0; i < A; i++) {
    float e = expf(logits[i] - maxv);
    out_probs[i] = e;
    sum += e;
  }

  if (sum <= 0.0f) {
    float u = 1.0f / (float)A;
    for (int i = 0; i < A; i++)
      out_probs[i] = u;
    return;
  }

  float inv = 1.0f / sum;
  for (int i = 0; i < A; i++)
    out_probs[i] *= inv;
}

void muzero_model_forward_batch(MuModel *m, const float *obs_batch, int B,
                                float *p_out, float *v_out) {
  if (!m || !obs_batch || !p_out || !v_out || B <= 0)
    return;

  int O = m->cfg.obs_dim;
  int L = m->cfg.latent_dim;
  int A = m->cfg.action_count;

  float *latent = (float *)malloc(sizeof(float) * L);
  float *logits = (float *)malloc(sizeof(float) * A);
  if (!latent || !logits) {
    free(latent);
    free(logits);
    return;
  }

  for (int i = 0; i < B; i++) {
    const float *obs = obs_batch + i * O;

    mu_model_repr(m, obs, latent);

    float value = 0.0f;
    mu_model_predict(m, latent, logits, &value);

    softmaxf(logits, A, p_out + i * A);
    v_out[i] = value; // already tanh()’d in mu_model_predict
  }

  free(latent);
  free(logits);
}

void muzero_model_train_batch(MuModel *m, const float *obs_batch,
                              const float *pi_batch, const float *z_batch,
                              int B, float lr) {
  if (!m || !obs_batch || !pi_batch || !z_batch || B <= 0)
    return;
  if (lr <= 0.0f)
    return;

  int O = m->cfg.obs_dim;
  int L = m->cfg.latent_dim;
  int A = m->cfg.action_count;

  // temporary buffers
  float *latent = (float *)malloc(sizeof(float) * L);
  float *preact = (float *)malloc(sizeof(float) * L); // repr pre-activation
  float *logits = (float *)malloc(sizeof(float) * A);
  float *probs = (float *)malloc(sizeof(float) * A);
  float *dlogits = (float *)malloc(sizeof(float) * A);
  float *dlatent = (float *)malloc(sizeof(float) * L);

  // gradients
  float *g_repr = (float *)calloc((size_t)(L * O), sizeof(float));
  float *g_pred = (float *)calloc((size_t)((A + 1) * L), sizeof(float));

  if (!latent || !preact || !logits || !probs || !dlogits || !dlatent ||
      !g_repr || !g_pred) {
    free(latent);
    free(preact);
    free(logits);
    free(probs);
    free(dlogits);
    free(dlatent);
    free(g_repr);
    free(g_pred);
    return;
  }

  // ----- accumulate grads over batch -----
  for (int i = 0; i < B; i++) {
    const float *obs = obs_batch + i * O;
    const float *pi_t = pi_batch + i * A;
    float z_t = z_batch[i];

    // representation forward with saved preact
    for (int li = 0; li < L; li++) {
      float s = 0.0f;
      for (int j = 0; j < O; j++)
        s += obs[j] * m->repr_W[li * O + j];
      preact[li] = s;
      latent[li] = tanhf(s);
    }

    // prediction forward
    float v = 0.0f;
    mu_model_predict(m, latent, logits, &v);
    softmaxf(logits, A, probs);

    // policy gradient: dL/dlogits = (p - pi_target)
    for (int a = 0; a < A; a++)
      dlogits[a] = (probs[a] - pi_t[a]);

    // value gradient (MSE): v = tanh(v_lin)
    // d/dv = 2*(v - z); dv/dv_lin = (1 - v^2)
    float dv_lin = 2.0f * (v - z_t) * (1.0f - v * v);

    // pred_W grads + dlatent from both heads
    for (int li = 0; li < L; li++)
      dlatent[li] = 0.0f;

    // policy head weights: pred_W[a*L + li]
    for (int a = 0; a < A; a++) {
      for (int li = 0; li < L; li++) {
        g_pred[a * L + li] += dlogits[a] * latent[li];
        dlatent[li] += dlogits[a] * m->pred_W[a * L + li];
      }
    }

    // value head weights start at (A*L)
    int value_off = A * L;
    for (int li = 0; li < L; li++) {
      g_pred[value_off + li] += dv_lin * latent[li];
      dlatent[li] += dv_lin * m->pred_W[value_off + li];
    }

    // backprop through tanh in representation
    for (int li = 0; li < L; li++) {
      float dpre = dlatent[li] * (1.0f - latent[li] * latent[li]); // tanh'
      for (int j = 0; j < O; j++) {
        g_repr[li * O + j] += dpre * obs[j];
      }
    }
  }

  // ----- SGD update (average over batch) -----
  float scale = lr / (float)B;

  // update repr_W
  for (int idx = 0; idx < L * O; idx++) {
    m->repr_W[idx] -= scale * g_repr[idx];
  }

  // update pred_W (policy + value head)
  for (int idx = 0; idx < (A + 1) * L; idx++) {
    m->pred_W[idx] -= scale * g_pred[idx];
  }

  // (dyn_W not trained by this loss, leave it unchanged)

  free(latent);
  free(preact);
  free(logits);
  free(probs);
  free(dlogits);
  free(dlatent);
  free(g_repr);
  free(g_pred);
}

void muzero_model_train_dynamics_batch(MuModel *m, const float *obs_batch,
                                       const int *a_batch, const float *r_batch,
                                       const float *next_obs_batch, int B,
                                       float lr, float *out_latent_mse,
                                       float *out_reward_mse) {
  if (out_latent_mse)
    *out_latent_mse = 0.0f;
  if (out_reward_mse)
    *out_reward_mse = 0.0f;

  if (!m || !obs_batch || !a_batch || !r_batch || !next_obs_batch || B <= 0)
    return;
  if (lr <= 0.0f)
    return;

  const int O = m->cfg.obs_dim;
  const int L = m->cfg.latent_dim;
  const int A = m->cfg.action_count;

  float *h = (float *)malloc(sizeof(float) * (size_t)L);
  float *h2 = (float *)malloc(sizeof(float) * (size_t)L);
  float *h_tgt = (float *)malloc(sizeof(float) * (size_t)L);
  float *pre2 = (float *)malloc(sizeof(float) * (size_t)L);

  float *g_dyn = (float *)calloc((size_t)L * (size_t)(L + 1), sizeof(float));
  float *g_rewW = (float *)calloc((size_t)L, sizeof(float));
  float g_rewB = 0.0f;

  if (!h || !h2 || !h_tgt || !pre2 || !g_dyn || !g_rewW) {
    free(h);
    free(h2);
    free(h_tgt);
    free(pre2);
    free(g_dyn);
    free(g_rewW);
    return;
  }

  float latent_mse_acc = 0.0f;
  float reward_mse_acc = 0.0f;

  for (int i = 0; i < B; i++) {
    const float *obs = obs_batch + (size_t)i * (size_t)O;
    const float *obs2 = next_obs_batch + (size_t)i * (size_t)O;
    const int act = a_batch[i];
    const float r_tgt = r_batch[i];

    // h = repr(s), h_tgt = repr(s')
    mu_model_repr(m, obs, h);
    mu_model_repr(m, obs2, h_tgt);

    // normalize action to [-1,1]
    float a = (A > 1) ? (float)act / (float)(A - 1) : 0.0f;
    float a2 = a * 2.0f - 1.0f;

    // forward dynamics: pre2 -> h2
    for (int li = 0; li < L; li++) {
      float sum = 0.0f;
      const float *row = &m->dyn_W[li * (L + 1)];
      for (int j = 0; j < L; j++)
        sum += h[j] * row[j];
      sum += a2 * row[L];
      pre2[li] = sum;
      h2[li] = tanhf(sum);
    }

    // reward head forward on h2
    float r_lin = m->rew_b;
    for (int li = 0; li < L; li++)
      r_lin += m->rew_W[li] * h2[li];
    float r_pred = tanhf(r_lin);

    // accumulate losses (for logging)
    // latent mse
    for (int li = 0; li < L; li++) {
      float d = h2[li] - h_tgt[li];
      latent_mse_acc += d * d;
    }
    // reward mse
    {
      float dr = (r_pred - r_tgt);
      reward_mse_acc += dr * dr;
    }

    // grads: reward mse through tanh
    float drlin = 2.0f * (r_pred - r_tgt) * (1.0f - r_pred * r_pred);

    g_rewB += drlin;
    for (int li = 0; li < L; li++)
      g_rewW[li] += drlin * h2[li];

    // backprop into dyn via h2
    for (int li = 0; li < L; li++) {
      float d_h2 = 2.0f * (h2[li] - h_tgt[li]) + drlin * m->rew_W[li];
      float d_pre = d_h2 * (1.0f - h2[li] * h2[li]);

      float *grow = &g_dyn[li * (L + 1)];
      for (int j = 0; j < L; j++)
        grow[j] += d_pre * h[j];
      grow[L] += d_pre * a2;
    }
  }

  // mean losses
  if (out_latent_mse)
    *out_latent_mse = latent_mse_acc / (float)(B * L);
  if (out_reward_mse)
    *out_reward_mse = reward_mse_acc / (float)B;

  // SGD update (avg)
  float scale = lr / (float)B;
  for (int idx = 0; idx < L * (L + 1); idx++)
    m->dyn_W[idx] -= scale * g_dyn[idx];
  for (int li = 0; li < L; li++)
    m->rew_W[li] -= scale * g_rewW[li];
  m->rew_b -= scale * g_rewB;

  free(h);
  free(h2);
  free(h_tgt);
  free(pre2);
  free(g_dyn);
  free(g_rewW);
}
