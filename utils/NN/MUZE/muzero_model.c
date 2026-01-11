#include "muzero_model.h"
#include "replay_buffer.h"
#include "runtime.h"
#include "trainer.h"
#include "../NN.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void nn_forward_raw(NN_t *nn, const float *in_f, int in_n, float *out_f,
                           long double **raw_out);
static void softmax_f(const float *logits, int n, float *out);

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

  float a = (m->cfg.action_count > 1)
                ? ((float)action / (float)(m->cfg.action_count - 1))
                : 0.0f;
  float a2 = a * 2.0f - 1.0f; // -1..1

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

static float value_transform(const MuModel *m, float v) {
  if (!m || !m->value_norm_enabled)
    return v;
  float eps = (m->value_rescale_eps > 0.0f) ? m->value_rescale_eps : 0.001f;
  float sign = (v >= 0.0f) ? 1.0f : -1.0f;
  float av = fabsf(v);
  return sign * (sqrtf(av + 1.0f) - 1.0f) + eps * v;
}

static float value_transform_inv(const MuModel *m, float v) {
  if (!m || !m->value_norm_enabled)
    return v;
  float eps = (m->value_rescale_eps > 0.0f) ? m->value_rescale_eps : 0.001f;
  float sign = (v >= 0.0f) ? 1.0f : -1.0f;
  float av = fabsf(v);
  float inner = 1.0f + 4.0f * eps * (av + 1.0f + eps);
  float sq = sqrtf(inner);
  float inv = ((sq - 1.0f) / (2.0f * eps)) - 1.0f;
  return sign * inv;
}

static void value_range_update(MuModel *m, float v) {
  (void)m;
  (void)v;
}

static float value_normalize(const MuModel *m, float v) {
  return value_transform(m, v);
}

float mu_model_denorm_value(MuModel *m, float v_norm) {
  return value_transform_inv(m, v_norm);
}

float mu_model_value_transform(MuModel *m, float v) {
  return value_transform(m, v);
}

float mu_model_value_transform_inv(MuModel *m, float v_norm) {
  return value_transform_inv(m, v_norm);
}

int mu_model_predict_value_support(MuModel *m, const float *latent,
                                   float *out_probs, int max_bins) {
  if (!m || !latent || !out_probs)
    return 0;
  if (!m->use_value_support || !m->use_nn || !m->nn_pred)
    return 0;
  int A = m->cfg.action_count;
  int bins = m->support_size;
  if (bins <= 0)
    return 0;
  if (max_bins < bins)
    bins = max_bins;

  float *tmp = (float *)malloc(sizeof(float) * (size_t)(A + m->support_size));
  if (!tmp)
    return 0;
  nn_forward_raw(m->nn_pred, latent, m->cfg.latent_dim, tmp, NULL);
  softmax_f(tmp + A, m->support_size, out_probs);
  if (bins < m->support_size) {
    float sum = 0.0f;
    for (int i = 0; i < bins; i++)
      sum += out_probs[i];
    if (sum > 0.0f) {
      float inv = 1.0f / sum;
      for (int i = 0; i < bins; i++)
        out_probs[i] *= inv;
    }
  }
  free(tmp);
  return bins;
}

float mu_model_support_expected(MuModel *m, const float *probs, int bins) {
  if (!m || !probs || bins <= 0)
    return 0.0f;
  if (bins == 1)
    return m->support_min;
  float delta = (m->support_max - m->support_min) / (float)(bins - 1);
  float sum = 0.0f;
  for (int i = 0; i < bins; i++) {
    float v = m->support_min + delta * (float)i;
    sum += probs[i] * v;
  }
  return sum;
}

/* ------------------------
   NN_t-backed model helpers
   ------------------------ */
static NN_t *nn_make_mlp(size_t in, size_t hidden, size_t out,
                         OptimizerType opt, LossFunctionType loss,
                         LossDerivativeType lossd, long double lr) {
  size_t layers[4];
  layers[0] = in;
  layers[1] = hidden;
  layers[2] = out;
  layers[3] = 0;

  ActivationFunctionType act[3] = {TANH, TANH, TANH};
  ActivationDerivativeType actd[3] = {TANH_DERIVATIVE, TANH_DERIVATIVE,
                                      TANH_DERIVATIVE};

  return NN_init(layers, act, actd, loss, lossd, L2, opt, lr);
}

static void nn_action_onehot(long double *dst, int A, int action) {
  for (int i = 0; i < A; i++)
    dst[i] = 0.0L;
  if (action >= 0 && action < A)
    dst[action] = 1.0L;
}

static int nn_copy_weights(NN_t *dst, const NN_t *src) {
  if (!dst || !src)
    return 0;
  if (dst->numLayers != src->numLayers)
    return 0;
  for (size_t i = 0; i < dst->numLayers; i++) {
    if (dst->layers[i] != src->layers[i])
      return 0;
  }

  for (size_t l = 0; l + 1 < dst->numLayers; l++) {
    size_t in_size = dst->layers[l];
    size_t out_size = dst->layers[l + 1];
    size_t wcount = in_size * out_size;
    memcpy(dst->weights[l], src->weights[l],
           sizeof(long double) * wcount);
    memcpy(dst->biases[l], src->biases[l],
           sizeof(long double) * out_size);
  }
  return 1;
}

static void clamp_ld(long double *v, int n, float clip) {
  if (!(clip > 0.0f))
    return;
  long double lo = -(long double)clip;
  long double hi = (long double)clip;
  for (int i = 0; i < n; i++) {
    if (v[i] < lo)
      v[i] = lo;
    else if (v[i] > hi)
      v[i] = hi;
  }
}

static void nn_forward_tanh(NN_t *nn, const float *in_f, int in_n,
                            float *out_f, long double **raw_out) {
  long double *in = (long double *)malloc(sizeof(long double) * (size_t)in_n);
  if (!in)
    return;
  for (int i = 0; i < in_n; i++)
    in[i] = (long double)in_f[i];

  long double *raw = NN_forward(nn, in);
  if (raw && out_f) {
    size_t out_n = nn->layers[nn->numLayers - 1];
    for (size_t i = 0; i < out_n; i++) {
      long double v = tanhl(raw[i]);
      out_f[i] = (float)v;
    }
  }

  if (raw_out)
    *raw_out = raw;
  else
    free(raw);
  free(in);
}

static void nn_forward_raw(NN_t *nn, const float *in_f, int in_n,
                           float *out_f, long double **raw_out) {
  long double *in = (long double *)malloc(sizeof(long double) * (size_t)in_n);
  if (!in)
    return;
  for (int i = 0; i < in_n; i++)
    in[i] = (long double)in_f[i];

  long double *raw = NN_forward(nn, in);
  if (raw && out_f) {
    size_t out_n = nn->layers[nn->numLayers - 1];
    for (size_t i = 0; i < out_n; i++)
      out_f[i] = (float)raw[i];
  }

  if (raw_out)
    *raw_out = raw;
  else
    free(raw);
  free(in);
}

static long double *ld_from_float(const float *in_f, int n) {
  long double *out = (long double *)malloc(sizeof(long double) * (size_t)n);
  if (!out)
    return NULL;
  for (int i = 0; i < n; i++)
    out[i] = (long double)in_f[i];
  return out;
}

static void softmax_ld(const long double *logits, int n, long double *out) {
  long double maxv = logits[0];
  for (int i = 1; i < n; i++) {
    if (logits[i] > maxv)
      maxv = logits[i];
  }
  long double sum = 0.0L;
  for (int i = 0; i < n; i++) {
    long double v = expl(logits[i] - maxv);
    out[i] = v;
    sum += v;
  }
  if (sum > 0.0L) {
    long double inv = 1.0L / sum;
    for (int i = 0; i < n; i++)
      out[i] *= inv;
  } else {
    long double u = (n > 0) ? (1.0L / (long double)n) : 0.0L;
    for (int i = 0; i < n; i++)
      out[i] = u;
  }
}

static void support_project(float v, int n, float vmin, float vmax,
                            float *out) {
  if (n <= 0 || !out)
    return;
  for (int i = 0; i < n; i++)
    out[i] = 0.0f;
  if (n == 1) {
    out[0] = 1.0f;
    return;
  }

  if (v < vmin)
    v = vmin;
  if (v > vmax)
    v = vmax;

  float delta = (vmax - vmin) / (float)(n - 1);
  if (delta <= 0.0f) {
    out[0] = 1.0f;
    return;
  }
  float idx = (v - vmin) / delta;
  int i0 = (int)floorf(idx);
  int i1 = i0 + 1;
  if (i0 < 0)
    i0 = 0;
  if (i0 >= n)
    i0 = n - 1;
  if (i1 < 0)
    i1 = 0;
  if (i1 >= n)
    i1 = n - 1;

  float frac = idx - (float)i0;
  if (i0 == i1) {
    out[i0] = 1.0f;
    return;
  }
  out[i0] = 1.0f - frac;
  out[i1] = frac;
}

static float support_expected(const long double *probs, int n, float vmin,
                              float vmax) {
  if (!probs || n <= 0)
    return 0.0f;
  if (n == 1)
    return vmin;
  float delta = (vmax - vmin) / (float)(n - 1);
  long double sum = 0.0L;
  for (int i = 0; i < n; i++) {
    float v = vmin + delta * (float)i;
    sum += probs[i] * (long double)v;
  }
  return (float)sum;
}

static void softmax_f(const float *logits, int n, float *out) {
  if (!logits || !out || n <= 0)
    return;
  float maxv = logits[0];
  for (int i = 1; i < n; i++)
    if (logits[i] > maxv)
      maxv = logits[i];
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    out[i] = expf(logits[i] - maxv);
    sum += out[i];
  }
  if (sum > 0.0f) {
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++)
      out[i] *= inv;
  } else {
    float u = 1.0f / (float)n;
    for (int i = 0; i < n; i++)
      out[i] = u;
  }
}

static void mu_model_train_policy_value_nn(
    MuModel *m, const float *obs_batch, const float *pi_batch,
    const float *z_batch, const float *weights, int B, float lr) {
  if (!m || !m->nn_repr || !m->nn_pred || !obs_batch || !pi_batch ||
      !z_batch || B <= 0)
    return;

  const int O = m->cfg.obs_dim;
  const int L = m->cfg.latent_dim;
  const int A = m->cfg.action_count;

  for (int i = 0; i < B; i++) {
    float w = weights ? weights[i] : 1.0f;
    const float *obs = obs_batch + (size_t)i * (size_t)O;
    const float *pi = pi_batch + (size_t)i * (size_t)A;
    float z_tgt = z_batch[i];

    long double *obs_ld = ld_from_float(obs, O);
    if (!obs_ld)
      continue;

    long double *repr_raw = NN_forward(m->nn_repr, obs_ld);
    if (!repr_raw) {
      free(obs_ld);
      continue;
    }

    float *latent = (float *)malloc(sizeof(float) * (size_t)L);
    if (!latent) {
      free(obs_ld);
      free(repr_raw);
      continue;
    }
    for (int j = 0; j < L; j++)
      latent[j] = (float)tanh((double)repr_raw[j]);

    long double *latent_ld = ld_from_float(latent, L);
    if (!latent_ld) {
      free(obs_ld);
      free(repr_raw);
      free(latent);
      continue;
    }

    long double *pred_raw = NN_forward(m->nn_pred, latent_ld);
    if (!pred_raw) {
      free(obs_ld);
      free(repr_raw);
      free(latent);
      free(latent_ld);
      continue;
    }

    int vout = m->use_value_support ? m->support_size : 1;
    long double *delta = (long double *)calloc((size_t)(A + vout),
                                               sizeof(long double));
    long double *probs = (long double *)malloc(sizeof(long double) * (size_t)A);
    if (!delta || !probs) {
      free(obs_ld);
      free(repr_raw);
      free(latent);
      free(latent_ld);
      free(pred_raw);
      free(delta);
      free(probs);
      continue;
    }

    softmax_ld(pred_raw, A, probs);
    for (int a = 0; a < A; a++)
      delta[a] = (long double)w * (long double)m->w_policy *
                 (probs[a] - (long double)pi[a]);

    if (m->use_value_support) {
      long double *v_logits =
          (long double *)malloc(sizeof(long double) * (size_t)vout);
      long double *v_probs =
          (long double *)malloc(sizeof(long double) * (size_t)vout);
      float *v_tgt = (float *)malloc(sizeof(float) * (size_t)vout);
      if (v_logits && v_probs && v_tgt) {
        for (int j = 0; j < vout; j++)
          v_logits[j] = pred_raw[A + j];
        softmax_ld(v_logits, vout, v_probs);
        support_project(z_tgt, vout, m->support_min, m->support_max, v_tgt);

        for (int j = 0; j < vout; j++) {
          delta[A + j] = (long double)w * (long double)m->w_value *
                         (v_probs[j] - (long double)v_tgt[j]);
        }
      }
      free(v_logits);
      free(v_probs);
      free(v_tgt);
    } else {
      long double v_raw = pred_raw[A];
      long double v_pred = tanhl(v_raw);
      long double dv = (long double)w * (long double)m->w_value * 2.0L *
                       (v_pred - (long double)z_tgt);
      delta[A] = dv * (1.0L - v_pred * v_pred);
    }

    clamp_ld(delta, A + vout, m->grad_clip);

    m->nn_pred->learningRate = (long double)lr;
    long double *grad_latent =
        NN_backprop_custom_delta_inputgrad(m->nn_pred, latent_ld, delta);

    if (grad_latent) {
      long double *delta_repr =
          (long double *)calloc((size_t)L, sizeof(long double));
      if (delta_repr) {
        for (int j = 0; j < L; j++) {
          long double h = (long double)latent[j];
          delta_repr[j] = grad_latent[j] * (1.0L - h * h);
        }
        m->nn_repr->learningRate = (long double)lr;
        NN_backprop_custom_delta(m->nn_repr, obs_ld, delta_repr);
        free(delta_repr);
      }
      free(grad_latent);
    }

    free(obs_ld);
    free(repr_raw);
    free(latent);
    free(latent_ld);
    free(pred_raw);
    free(delta);
    free(probs);
  }
}

static void mu_model_train_dynamics_nn(
    MuModel *m, const float *obs_batch, const int *a_batch,
    const float *r_batch, const float *next_obs_batch, const float *weights,
    int train_reward_head, int B, float lr, float *out_latent_mse,
    float *out_reward_mse) {
  if (out_latent_mse)
    *out_latent_mse = 0.0f;
  if (out_reward_mse)
    *out_reward_mse = 0.0f;

  if (!m || !m->nn_repr || !m->nn_dyn || !obs_batch || !a_batch ||
      !r_batch || !next_obs_batch || B <= 0)
    return;

  const int O = m->cfg.obs_dim;
  const int L = m->cfg.latent_dim;
  const int A = m->cfg.action_count;

  double lat_mse = 0.0;
  double rew_mse = 0.0;

  for (int i = 0; i < B; i++) {
    float w = weights ? weights[i] : 1.0f;
    const float *obs = obs_batch + (size_t)i * (size_t)O;
    const float *next_obs = next_obs_batch + (size_t)i * (size_t)O;
    int action = a_batch[i];
    float r_tgt = r_batch[i];

    long double *obs_ld = ld_from_float(obs, O);
    long double *next_obs_ld = ld_from_float(next_obs, O);
    if (!obs_ld || !next_obs_ld) {
      free(obs_ld);
      free(next_obs_ld);
      continue;
    }

    long double *h_raw = NN_forward(m->nn_repr, obs_ld);
    long double *h_next_raw = NN_forward(m->nn_repr, next_obs_ld);
    if (!h_raw || !h_next_raw) {
      free(obs_ld);
      free(next_obs_ld);
      free(h_raw);
      free(h_next_raw);
      continue;
    }

    float *h = (float *)malloc(sizeof(float) * (size_t)L);
    float *h_next = (float *)malloc(sizeof(float) * (size_t)L);
    if (!h || !h_next) {
      free(obs_ld);
      free(next_obs_ld);
      free(h_raw);
      free(h_next_raw);
      free(h);
      free(h_next);
      continue;
    }
    for (int j = 0; j < L; j++) {
      h[j] = (float)tanh((double)h_raw[j]);
      h_next[j] = (float)tanh((double)h_next_raw[j]);
    }

    float *dyn_in_f =
        (float *)malloc(sizeof(float) * (size_t)(L + m->action_embed_dim));
    if (!dyn_in_f) {
      free(obs_ld);
      free(next_obs_ld);
      free(h_raw);
      free(h_next_raw);
      free(h);
      free(h_next);
      continue;
    }
    memcpy(dyn_in_f, h, sizeof(float) * (size_t)L);
    if (m->action_embed && action >= 0 && action < A) {
      const float *emb =
          m->action_embed + (size_t)action * (size_t)m->action_embed_dim;
      memcpy(dyn_in_f + L, emb,
             sizeof(float) * (size_t)m->action_embed_dim);
    } else {
      memset(dyn_in_f + L, 0,
             sizeof(float) * (size_t)m->action_embed_dim);
    }

    long double *dyn_in_ld =
        ld_from_float(dyn_in_f, L + m->action_embed_dim);
    long double *h_pred_raw = NN_forward(m->nn_dyn, dyn_in_ld);
    if (!dyn_in_ld || !h_pred_raw) {
      free(obs_ld);
      free(next_obs_ld);
      free(h_raw);
      free(h_next_raw);
      free(h);
      free(h_next);
      free(dyn_in_f);
      free(dyn_in_ld);
      free(h_pred_raw);
      continue;
    }

    long double *delta_dyn =
        (long double *)calloc((size_t)L, sizeof(long double));
    if (!delta_dyn) {
      free(obs_ld);
      free(next_obs_ld);
      free(h_raw);
      free(h_next_raw);
      free(h);
      free(h_next);
      free(dyn_in_f);
      free(dyn_in_ld);
      free(h_pred_raw);
      continue;
    }

    for (int j = 0; j < L; j++) {
      long double h_pred = tanhl(h_pred_raw[j]);
      long double d = h_pred - (long double)h_next[j];
      lat_mse += (double)(d * d);
      long double dd = (long double)w * (long double)m->w_latent * 2.0L * d;
      delta_dyn[j] = dd * (1.0L - h_pred * h_pred);
    }

    clamp_ld(delta_dyn, L, m->grad_clip);
    m->nn_dyn->learningRate = (long double)lr;
    NN_backprop_custom_delta(m->nn_dyn, dyn_in_ld, delta_dyn);

    if (train_reward_head && m->nn_reward) {
      long double *h_pred_ld = (long double *)malloc(sizeof(long double) *
                                                     (size_t)L);
      if (h_pred_ld) {
        for (int j = 0; j < L; j++)
          h_pred_ld[j] = tanhl(h_pred_raw[j]);

        long double *rew_raw = NN_forward(m->nn_reward, h_pred_ld);
        if (rew_raw) {
          if (m->use_reward_support) {
            int rout = m->support_size;
            long double *r_logits =
                (long double *)malloc(sizeof(long double) * (size_t)rout);
            long double *r_probs =
                (long double *)malloc(sizeof(long double) * (size_t)rout);
            long double *r_delta =
                (long double *)malloc(sizeof(long double) * (size_t)rout);
            float *r_dist = (float *)malloc(sizeof(float) * (size_t)rout);
            if (r_logits && r_probs && r_delta && r_dist) {
              for (int j = 0; j < rout; j++)
                r_logits[j] = rew_raw[j];
              softmax_ld(r_logits, rout, r_probs);
              {
                float r_pred = support_expected(r_probs, rout, m->support_min,
                                                m->support_max);
                float d = r_pred - r_tgt;
                rew_mse += (double)(d * d);
              }
              support_project(r_tgt, rout, m->support_min, m->support_max,
                              r_dist);
              for (int j = 0; j < rout; j++) {
                r_delta[j] = (long double)w * (long double)m->w_reward *
                             (r_probs[j] - (long double)r_dist[j]);
              }
              clamp_ld(r_delta, rout, m->grad_clip);
              m->nn_reward->learningRate = (long double)lr;
              NN_backprop_custom_delta(m->nn_reward, h_pred_ld, r_delta);
            }
            free(r_logits);
            free(r_probs);
            free(r_delta);
            free(r_dist);
          } else {
            long double r_pred = tanhl(rew_raw[0]);
            long double d = r_pred - (long double)r_tgt;
            rew_mse += (double)(d * d);
            long double dr = (long double)w * (long double)m->w_reward * 2.0L * d;
            long double delta_rew = dr * (1.0L - r_pred * r_pred);
            clamp_ld(&delta_rew, 1, m->grad_clip);
            m->nn_reward->learningRate = (long double)lr;
            NN_backprop_custom_delta(m->nn_reward, h_pred_ld, &delta_rew);
          }
          free(rew_raw);
        }
        free(h_pred_ld);
      }
    }

    free(obs_ld);
    free(next_obs_ld);
    free(h_raw);
    free(h_next_raw);
    free(h);
    free(h_next);
    free(dyn_in_f);
    free(dyn_in_ld);
    free(h_pred_raw);
    free(delta_dyn);
  }

  if (out_latent_mse)
    *out_latent_mse = (float)(lat_mse / (double)(B * L));
  if (out_reward_mse)
    *out_reward_mse = (float)(rew_mse / (double)B);
}

static void mu_model_train_unroll_nn(
    MuModel *m, const float *obs_seq, const float *pi_seq,
    const float *z_seq, const float *vprefix_seq, const int *a_seq,
    const float *r_seq, const int *done_seq, const float *weights, int B,
    int unroll_steps, int bootstrap_steps, float discount, float lr,
    float *out_policy_loss, float *out_value_loss, float *out_reward_loss,
    float *out_latent_loss) {
  if (out_policy_loss)
    *out_policy_loss = 0.0f;
  if (out_value_loss)
    *out_value_loss = 0.0f;
  if (out_reward_loss)
    *out_reward_loss = 0.0f;
  if (out_latent_loss)
    *out_latent_loss = 0.0f;

  if (!m || !m->nn_repr || !m->nn_dyn || !m->nn_pred || !m->nn_vprefix ||
      !obs_seq || !pi_seq || !z_seq || !a_seq || !r_seq || !done_seq || B <= 0)
    return;

  const int O = m->cfg.obs_dim;
  const int L = m->cfg.latent_dim;
  const int A = m->cfg.action_count;
  const int K = unroll_steps;
  if (bootstrap_steps <= 0)
    bootstrap_steps = 1;
  if (!(discount > 0.0f))
    discount = 0.997f;

  size_t steps = (size_t)K + 1;

  double pol_acc = 0.0;
  double val_acc = 0.0;
  double rew_acc = 0.0;
  double lat_acc = 0.0;
  double pol_cnt = 0.0;
  double val_cnt = 0.0;
  double rew_cnt = 0.0;
  double lat_cnt = 0.0;

  for (int b = 0; b < B; b++) {
    float w = weights ? weights[b] : 1.0f;
    const float *obs0 = obs_seq + (size_t)b * steps * (size_t)O;
    const float *pi0 = pi_seq + (size_t)b * steps * (size_t)A;
    const float *z0 = z_seq + (size_t)b * steps;
    const float *vp0 = vprefix_seq ? (vprefix_seq + (size_t)b * steps) : NULL;
    const int *a0 = a_seq + (size_t)b * (size_t)K;
    const float *r0 = r_seq + (size_t)b * (size_t)K;
    const int *d0 = done_seq + (size_t)b * (size_t)K;

    float *h = (float *)malloc(sizeof(float) * steps * (size_t)L);
    float *h_tgt = (float *)malloc(sizeof(float) * steps * (size_t)L);
    long double *dh =
        (long double *)calloc(steps * (size_t)L, sizeof(long double));
    float *mask = (float *)malloc(sizeof(float) * steps);
    if (!h || !h_tgt || !dh || !mask) {
      free(h);
      free(h_tgt);
      free(dh);
      free(mask);
      continue;
    }

    mask[0] = 1.0f;
    for (int k = 0; k < K; k++)
      mask[k + 1] = (mask[k] > 0.0f && d0[k] == 0) ? 1.0f : 0.0f;

    long double *obs_ld = ld_from_float(obs0, O);
    if (!obs_ld) {
      free(h);
      free(h_tgt);
      free(dh);
      free(mask);
      continue;
    }
    long double *h_raw = NN_forward(m->nn_repr, obs_ld);
    if (!h_raw) {
      free(obs_ld);
      free(h);
      free(h_tgt);
      free(dh);
      free(mask);
      continue;
    }
    for (int j = 0; j < L; j++)
      h[j] = (float)tanh((double)h_raw[j]);
    free(obs_ld);
    free(h_raw);

    for (int k = 0; k < K; k++) {
      float *h_k = h + (size_t)k * (size_t)L;
      float *h_k1 = h + (size_t)(k + 1) * (size_t)L;

      float *dyn_in =
          (float *)malloc(sizeof(float) * (size_t)(L + m->action_embed_dim));
      if (!dyn_in)
        break;
      memcpy(dyn_in, h_k, sizeof(float) * (size_t)L);
      if (m->action_embed && a0[k] >= 0 && a0[k] < A) {
        const float *emb =
            m->action_embed + (size_t)a0[k] * (size_t)m->action_embed_dim;
        memcpy(dyn_in + L, emb,
               sizeof(float) * (size_t)m->action_embed_dim);
      } else {
        memset(dyn_in + L, 0,
               sizeof(float) * (size_t)m->action_embed_dim);
      }

      long double *dyn_in_ld =
          ld_from_float(dyn_in, L + m->action_embed_dim);
      long double *h_k1_raw = dyn_in_ld ? NN_forward(m->nn_dyn, dyn_in_ld)
                                        : NULL;
      if (!dyn_in_ld || !h_k1_raw) {
        free(dyn_in);
        free(dyn_in_ld);
        free(h_k1_raw);
        break;
      }

      for (int j = 0; j < L; j++)
        h_k1[j] = (float)tanh((double)h_k1_raw[j]);

      free(dyn_in);
      free(dyn_in_ld);
      free(h_k1_raw);
    }

    for (int k = 1; k <= K; k++) {
      const float *obs_k = obs0 + (size_t)k * (size_t)O;
      long double *obs_k_ld = ld_from_float(obs_k, O);
      long double *raw_k = obs_k_ld ? NN_forward(m->nn_repr, obs_k_ld) : NULL;
      if (obs_k_ld && raw_k) {
        for (int j = 0; j < L; j++)
          h_tgt[(size_t)k * (size_t)L + (size_t)j] =
              (float)tanh((double)raw_k[j]);
      }
      free(obs_k_ld);
      free(raw_k);
    }

    long double *v_pred = (long double *)malloc(sizeof(long double) * steps);
    if (!v_pred) {
      free(h);
      free(h_tgt);
      free(dh);
      free(mask);
      continue;
    }

    for (int k = 0; k <= K; k++) {
      if (mask[k] <= 0.0f)
        continue;

      float *h_k = h + (size_t)k * (size_t)L;
      long double *h_k_ld = ld_from_float(h_k, L);
      if (!h_k_ld)
        continue;
      long double *pred_raw = NN_forward(m->nn_pred, h_k_ld);
      if (!pred_raw) {
        free(h_k_ld);
        continue;
      }

      int vout = m->use_value_support ? m->support_size : 1;
      long double *delta = (long double *)calloc((size_t)(A + vout),
                                                 sizeof(long double));
      long double *probs =
          (long double *)malloc(sizeof(long double) * (size_t)A);
      if (!delta || !probs) {
        free(h_k_ld);
        free(pred_raw);
        free(delta);
        free(probs);
        continue;
      }

      softmax_ld(pred_raw, A, probs);
      for (int a = 0; a < A; a++) {
        long double d = probs[a] - (long double)pi0[(size_t)k * (size_t)A + a];
        delta[a] = (long double)w * (long double)m->w_policy * d;
        pol_acc += (double)(-(long double)pi0[(size_t)k * (size_t)A + a] *
                            logl(probs[a] + 1e-12L)) *
                   (double)w * (double)m->w_policy;
      }
      pol_cnt += 1.0;

      long double v = 0.0L;
      if (m->use_value_support) {
        long double *v_logits =
            (long double *)malloc(sizeof(long double) * (size_t)vout);
        long double *v_probs =
            (long double *)malloc(sizeof(long double) * (size_t)vout);
        if (v_logits && v_probs) {
          for (int j = 0; j < vout; j++)
            v_logits[j] = pred_raw[A + j];
          softmax_ld(v_logits, vout, v_probs);
          v = (long double)support_expected(v_probs, vout, m->support_min,
                                            m->support_max);
          v_pred[k] = v;
        }
        free(v_logits);
        free(v_probs);
      } else {
        long double v_raw = pred_raw[A];
        v = tanhl(v_raw);
        v_pred[k] = v;
      }

      long double target_v = (long double)z0[k];
      if (bootstrap_steps > 0) {
        long double G = 0.0L;
        long double gamma_pow = 1.0L;
        int done_flag = 0;
        int max_i = bootstrap_steps;
        for (int i = 0; i < max_i; i++) {
          int t = k + i;
          if (t >= K)
            break;
          G += gamma_pow * (long double)r0[t];
          gamma_pow *= (long double)discount;
          if (d0[t]) {
            done_flag = 1;
            break;
          }
        }
        long double bootstrap = 0.0L;
        if (!done_flag) {
          int t_boot = k + max_i;
          if (t_boot > K)
            t_boot = K;
          bootstrap = (long double)mu_model_denorm_value(
              m, (float)v_pred[t_boot]);
        }
        target_v = G + gamma_pow * bootstrap;
        target_v = (long double)mu_model_value_transform(m, (float)target_v);
      }

      if (m->use_value_support) {
        float *v_dist = (float *)malloc(sizeof(float) * (size_t)vout);
        long double *v_logits =
            (long double *)malloc(sizeof(long double) * (size_t)vout);
        long double *v_probs =
            (long double *)malloc(sizeof(long double) * (size_t)vout);
        if (v_dist && v_logits && v_probs) {
          support_project((float)target_v, vout, m->support_min,
                          m->support_max, v_dist);
          for (int j = 0; j < vout; j++)
            v_logits[j] = pred_raw[A + j];
          softmax_ld(v_logits, vout, v_probs);
          for (int j = 0; j < vout; j++)
            delta[A + j] =
                (long double)w * (long double)m->w_value *
                (v_probs[j] - (long double)v_dist[j]);
          val_acc += (double)(v - target_v) * (double)(v - target_v) *
                     (double)w * (double)m->w_value;
        }
        free(v_dist);
        free(v_logits);
        free(v_probs);
      } else {
        long double dv = (long double)w * (long double)m->w_value * 2.0L *
                         (v - target_v);
        delta[A] = dv * (1.0L - v * v);
        val_acc += (double)(v - target_v) * (double)(v - target_v) *
                   (double)w * (double)m->w_value;
      }
      val_cnt += 1.0;

      clamp_ld(delta, A + vout, m->grad_clip);
      m->nn_pred->learningRate = (long double)lr;
      long double *grad_h =
          NN_backprop_custom_delta_inputgrad(m->nn_pred, h_k_ld, delta);
      if (grad_h) {
        for (int j = 0; j < L; j++)
          dh[(size_t)k * (size_t)L + (size_t)j] += grad_h[j];
        free(grad_h);
      }

      free(h_k_ld);
      free(pred_raw);
      free(delta);
      free(probs);
    }

    for (int k = 0; k < K; k++) {
      if (mask[k] <= 0.0f)
        continue;

      float *h_k1 = h + (size_t)(k + 1) * (size_t)L;
      long double *h_k1_ld = ld_from_float(h_k1, L);
      if (!h_k1_ld)
        continue;

      long double *vp_raw = NN_forward(m->nn_vprefix, h_k1_ld);
      if (!vp_raw) {
        free(h_k1_ld);
        continue;
      }

      long double vp_tgt = (long double)r0[k];
      if (vp0)
        vp_tgt = (long double)vp0[k];
      else {
        long double prefix = 0.0L;
        long double gamma_pow = 1.0L;
        for (int i = 0; i <= k; i++) {
          prefix += gamma_pow * (long double)r0[i];
          gamma_pow *= (long double)discount;
        }
        vp_tgt = prefix;
      }

      if (m->use_reward_support) {
        int rout = m->support_size;
        long double *vp_logits =
            (long double *)malloc(sizeof(long double) * (size_t)rout);
        long double *vp_probs =
            (long double *)malloc(sizeof(long double) * (size_t)rout);
        long double *vp_delta =
            (long double *)malloc(sizeof(long double) * (size_t)rout);
        float *vp_dist = (float *)malloc(sizeof(float) * (size_t)rout);
        if (vp_logits && vp_probs && vp_delta && vp_dist) {
          for (int j = 0; j < rout; j++)
            vp_logits[j] = vp_raw[j];
          softmax_ld(vp_logits, rout, vp_probs);
          support_project((float)vp_tgt, rout, m->support_min, m->support_max,
                          vp_dist);
          {
            double ce = 0.0;
            for (int j = 0; j < rout; j++) {
              ce -= (double)vp_dist[j] * log((double)vp_probs[j] + 1e-12);
            }
            rew_acc += ce * (double)w * (double)m->w_vprefix;
            rew_cnt += 1.0;
          }
          for (int j = 0; j < rout; j++)
            vp_delta[j] =
                (long double)w * (long double)m->w_vprefix *
                (vp_probs[j] - (long double)vp_dist[j]);
          clamp_ld(vp_delta, rout, m->grad_clip);
          m->nn_vprefix->learningRate = (long double)lr;
          long double *grad_h =
              NN_backprop_custom_delta_inputgrad(m->nn_vprefix, h_k1_ld,
                                                 vp_delta);
          if (grad_h) {
            for (int j = 0; j < L; j++)
              dh[(size_t)(k + 1) * (size_t)L + (size_t)j] += grad_h[j];
            free(grad_h);
          }
        }
        free(vp_logits);
        free(vp_probs);
        free(vp_delta);
        free(vp_dist);
      } else {
        long double vp_pred = tanhl(vp_raw[0]);
        long double dvp =
            (long double)w * (long double)m->w_vprefix * 2.0L *
            (vp_pred - vp_tgt);
        long double delta_vp = dvp * (1.0L - vp_pred * vp_pred);
        clamp_ld(&delta_vp, 1, m->grad_clip);
        rew_acc += (double)(vp_pred - vp_tgt) * (double)(vp_pred - vp_tgt) *
                   (double)w * (double)m->w_vprefix;
        rew_cnt += 1.0;

        m->nn_vprefix->learningRate = (long double)lr;
        long double *grad_h =
            NN_backprop_custom_delta_inputgrad(m->nn_vprefix, h_k1_ld,
                                               &delta_vp);
        if (grad_h) {
          for (int j = 0; j < L; j++)
            dh[(size_t)(k + 1) * (size_t)L + (size_t)j] += grad_h[j];
          free(grad_h);
        }
      }

      if (mask[k + 1] > 0.0f) {
        float *h_t = h_tgt + (size_t)(k + 1) * (size_t)L;
        for (int j = 0; j < L; j++) {
          long double d = (long double)h_k1[j] - (long double)h_t[j];
          dh[(size_t)(k + 1) * (size_t)L + (size_t)j] +=
              (long double)w * (long double)m->w_latent * 2.0L * d;
          lat_acc +=
              (double)(d * d) * (double)w * (double)m->w_latent;
          lat_cnt += 1.0;
        }
      }

      free(h_k1_ld);
      free(vp_raw);
    }

    for (int k = K - 1; k >= 0; k--) {
      float *h_k = h + (size_t)k * (size_t)L;
      float *h_k1 = h + (size_t)(k + 1) * (size_t)L;

      long double *delta_dyn =
          (long double *)calloc((size_t)L, sizeof(long double));
      if (!delta_dyn)
        continue;
      for (int j = 0; j < L; j++) {
        long double dh_j = dh[(size_t)(k + 1) * (size_t)L + (size_t)j];
        long double hk1 = (long double)h_k1[j];
        delta_dyn[j] = dh_j * (1.0L - hk1 * hk1);
      }
      clamp_ld(delta_dyn, L, m->grad_clip);

      float *dyn_in =
          (float *)malloc(sizeof(float) * (size_t)(L + m->action_embed_dim));
      if (!dyn_in) {
        free(delta_dyn);
        continue;
      }
      memcpy(dyn_in, h_k, sizeof(float) * (size_t)L);
      if (m->action_embed && a0[k] >= 0 && a0[k] < A) {
        const float *emb =
            m->action_embed + (size_t)a0[k] * (size_t)m->action_embed_dim;
        memcpy(dyn_in + L, emb,
               sizeof(float) * (size_t)m->action_embed_dim);
      } else {
        memset(dyn_in + L, 0,
               sizeof(float) * (size_t)m->action_embed_dim);
      }

      long double *dyn_in_ld =
          ld_from_float(dyn_in, L + m->action_embed_dim);
      if (dyn_in_ld) {
        m->nn_dyn->learningRate = (long double)lr;
        long double *grad_in =
            NN_backprop_custom_delta_inputgrad(m->nn_dyn, dyn_in_ld,
                                               delta_dyn);
        if (grad_in) {
          for (int j = 0; j < L; j++)
            dh[(size_t)k * (size_t)L + (size_t)j] += grad_in[j];
          free(grad_in);
        }
      }
      free(dyn_in);
      free(dyn_in_ld);
      free(delta_dyn);
    }

    long double *obs0_ld = ld_from_float(obs0, O);
    if (obs0_ld) {
      long double *delta_repr =
          (long double *)calloc((size_t)L, sizeof(long double));
      if (delta_repr) {
        for (int j = 0; j < L; j++) {
          long double h0 = (long double)h[j];
          delta_repr[j] = dh[j] * (1.0L - h0 * h0);
        }
        clamp_ld(delta_repr, L, m->grad_clip);
        m->nn_repr->learningRate = (long double)lr;
        NN_backprop_custom_delta(m->nn_repr, obs0_ld, delta_repr);
        free(delta_repr);
      }
      free(obs0_ld);
    }

    free(h);
    free(h_tgt);
    free(dh);
    free(mask);
    free(v_pred);
  }

  if (out_policy_loss && pol_cnt > 0.0)
    *out_policy_loss = (float)(pol_acc / pol_cnt);
  if (out_value_loss && val_cnt > 0.0)
    *out_value_loss = (float)(val_acc / val_cnt);
  if (out_reward_loss && rew_cnt > 0.0)
    *out_reward_loss = (float)(rew_acc / rew_cnt);
  if (out_latent_loss && lat_cnt > 0.0)
    *out_latent_loss = (float)(lat_acc / lat_cnt);
}

/* ------------------------
   Create model
   ------------------------ */
MuModel *mu_model_create(const MuConfig *cfg) {
  MuModel *m = (MuModel *)calloc(1, sizeof(MuModel));
  if (!m)
    return NULL;
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

  m->vprefix_W_count = lat;
  m->vprefix_W = (float *)malloc(sizeof(float) * lat);
  m->vprefix_b = 0.0f;
  for (int i = 0; i < lat; i++)
    m->vprefix_W[i] = 0.01f;

  m->value_norm_enabled = 1;
  m->value_min = -1.0f;
  m->value_max = 1.0f;
  m->value_rescale_eps = 0.001f;

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
  m->train_policy_value = NULL;
  m->train_dynamics = NULL;
  m->train_unroll = NULL;
  m->use_nn = 0;

  return m;
}

static void mu_model_repr_nn(MuModel *m, const float *obs, float *latent_out) {
  if (!m || !m->nn_repr || !obs || !latent_out)
    return;
  nn_forward_tanh(m->nn_repr, obs, m->cfg.obs_dim, latent_out, NULL);
}

static void mu_model_predict_nn(MuModel *m, const float *latent,
                                float *policy_logits, float *value_out) {
  if (!m || !m->nn_pred || !latent || !policy_logits || !value_out)
    return;

  int A = m->cfg.action_count;
  int vout = m->use_value_support ? m->support_size : 1;
  float *tmp =
      (float *)malloc(sizeof(float) * (size_t)(A + vout));
  if (!tmp)
    return;

  nn_forward_raw(m->nn_pred, latent, m->cfg.latent_dim, tmp, NULL);
  for (int i = 0; i < A; i++)
    policy_logits[i] = tmp[i];
  if (m->use_value_support) {
    long double *v_logits =
        (long double *)malloc(sizeof(long double) * (size_t)vout);
    long double *v_probs =
        (long double *)malloc(sizeof(long double) * (size_t)vout);
    if (v_logits && v_probs) {
      for (int i = 0; i < vout; i++)
        v_logits[i] = (long double)tmp[A + i];
      softmax_ld(v_logits, vout, v_probs);
      *value_out =
          support_expected(v_probs, vout, m->support_min, m->support_max);
    } else {
      *value_out = 0.0f;
    }
    free(v_logits);
    free(v_probs);
  } else {
    *value_out = tanhf(tmp[A]);
  }
  free(tmp);
}

static void mu_model_dynamics_nn(MuModel *m, const float *latent_in, int action,
                                 float *latent_out, float *reward_out) {
  if (!m || !m->nn_dyn || !latent_in || !latent_out || !reward_out)
    return;

  int L = m->cfg.latent_dim;
  int E = m->action_embed_dim;
  int in_n = L + E;
  float *in = (float *)malloc(sizeof(float) * (size_t)in_n);
  if (!in)
    return;

  memcpy(in, latent_in, sizeof(float) * (size_t)L);
  if (m->action_embed && action >= 0 && action < m->cfg.action_count) {
    const float *emb =
        m->action_embed + (size_t)action * (size_t)E;
    memcpy(in + L, emb, sizeof(float) * (size_t)E);
  } else {
    memset(in + L, 0, sizeof(float) * (size_t)E);
  }

  nn_forward_tanh(m->nn_dyn, in, in_n, latent_out, NULL);

  if (m->nn_reward) {
    int rout = m->use_reward_support ? m->support_size : 1;
    float *r_raw = (float *)malloc(sizeof(float) * (size_t)rout);
    if (r_raw) {
      nn_forward_raw(m->nn_reward, latent_out, L, r_raw, NULL);
      if (m->use_reward_support) {
        long double *r_logits =
            (long double *)malloc(sizeof(long double) * (size_t)rout);
        long double *r_probs =
            (long double *)malloc(sizeof(long double) * (size_t)rout);
        if (r_logits && r_probs) {
          for (int i = 0; i < rout; i++)
            r_logits[i] = (long double)r_raw[i];
          softmax_ld(r_logits, rout, r_probs);
          *reward_out =
              support_expected(r_probs, rout, m->support_min, m->support_max);
        } else {
          *reward_out = 0.0f;
        }
        free(r_logits);
        free(r_probs);
      } else {
        *reward_out = tanhf(r_raw[0]);
      }
      free(r_raw);
    } else {
      *reward_out = 0.0f;
    }
  } else {
    *reward_out = 0.0f;
  }

  free(in);
}

static MuNNConfig mu_nn_default_cfg(const MuConfig *cfg) {
  MuNNConfig c;
  c.opt_repr = ADAM;
  c.opt_dyn = ADAM;
  c.opt_pred = ADAM;
  c.opt_vprefix = ADAM;
  c.opt_reward = ADAM;
  c.loss_repr = MSE;
  c.loss_dyn = MSE;
  c.loss_pred = MSE;
  c.loss_vprefix = MSE;
  c.loss_reward = MSE;
  c.lossd_repr = MSE_DERIVATIVE;
  c.lossd_dyn = MSE_DERIVATIVE;
  c.lossd_pred = MSE_DERIVATIVE;
  c.lossd_vprefix = MSE_DERIVATIVE;
  c.lossd_reward = MSE_DERIVATIVE;
  c.lr_repr = 0.001L;
  c.lr_dyn = 0.001L;
  c.lr_pred = 0.001L;
  c.lr_vprefix = 0.001L;
  c.lr_reward = 0.001L;
  c.hidden_repr = cfg ? (size_t)cfg->latent_dim : 32;
  c.hidden_dyn = cfg ? (size_t)cfg->latent_dim : 32;
  c.hidden_pred = cfg ? (size_t)cfg->latent_dim : 32;
  c.hidden_vprefix = cfg ? (size_t)cfg->latent_dim : 32;
  c.hidden_reward = cfg ? (size_t)cfg->latent_dim : 32;

  c.use_value_support = 1;
  c.use_reward_support = 1;
  c.support_size = 21;
  c.support_min = -2.0f;
  c.support_max = 2.0f;

  c.action_embed_dim = cfg ? cfg->latent_dim : 32;

  c.w_policy = 1.0f;
  c.w_value = 1.0f;
  c.w_vprefix = 1.0f;
  c.w_latent = 1.0f;
  c.w_reward = 1.0f;

  c.grad_clip = 5.0f;
  return c;
}

MuModel *mu_model_create_nn_with_cfg(const MuConfig *cfg,
                                     const MuNNConfig *nn_cfg) {
  if (!cfg)
    return NULL;
  MuModel *m = (MuModel *)calloc(1, sizeof(MuModel));
  if (!m)
    return NULL;
  m->cfg = *cfg;

  m->repr_W = NULL;
  m->dyn_W = NULL;
  m->pred_W = NULL;
  m->rew_W = NULL;
  m->vprefix_W = NULL;
  m->repr_W_count = 0;
  m->dyn_W_count = 0;
  m->pred_W_count = 0;
  m->rew_W_count = 0;
  m->vprefix_W_count = 0;

  m->value_norm_enabled = 1;
  m->value_min = -1.0f;
  m->value_max = 1.0f;
  m->value_rescale_eps = 0.001f;

  MuNNConfig c = nn_cfg ? *nn_cfg : mu_nn_default_cfg(cfg);

  if (c.support_size <= 0)
    c.support_size = 1;
  if (c.support_min >= c.support_max) {
    c.support_min = -2.0f;
    c.support_max = 2.0f;
  }
  if (c.action_embed_dim <= 0)
    c.action_embed_dim = cfg->latent_dim;

  m->use_value_support = c.use_value_support ? 1 : 0;
  m->use_reward_support = c.use_reward_support ? 1 : 0;
  m->support_size = c.support_size;
  m->support_min = c.support_min;
  m->support_max = c.support_max;
  m->action_embed_dim = c.action_embed_dim;
  m->action_embed_count = (int)((size_t)cfg->action_count *
                                (size_t)m->action_embed_dim);
  m->action_embed =
      (float *)malloc(sizeof(float) * (size_t)m->action_embed_count);
  if (!m->action_embed) {
    mu_model_free(m);
    return NULL;
  }
  for (int i = 0; i < m->action_embed_count; i++) {
    float r = (float)rand() / (float)RAND_MAX;
    m->action_embed[i] = (r - 0.5f) * 0.02f;
  }

  m->w_policy = c.w_policy;
  m->w_value = c.w_value;
  m->w_vprefix = c.w_vprefix;
  m->w_latent = c.w_latent;
  m->w_reward = c.w_reward;
  m->grad_clip = c.grad_clip;

  const size_t O = (size_t)cfg->obs_dim;
  const size_t L = (size_t)cfg->latent_dim;
  const size_t A = (size_t)cfg->action_count;
  const size_t vout = m->use_value_support ? (size_t)m->support_size : 1;
  const size_t rout = m->use_reward_support ? (size_t)m->support_size : 1;

  m->nn_repr = nn_make_mlp(O, c.hidden_repr, L, c.opt_repr, c.loss_repr,
                           c.lossd_repr, c.lr_repr);
  m->nn_dyn = nn_make_mlp(L + (size_t)m->action_embed_dim, c.hidden_dyn, L,
                          c.opt_dyn, c.loss_dyn,
                          c.lossd_dyn, c.lr_dyn);
  m->nn_pred = nn_make_mlp(L, c.hidden_pred, A + vout, c.opt_pred, c.loss_pred,
                           c.lossd_pred, c.lr_pred);
  m->nn_vprefix = nn_make_mlp(L, c.hidden_vprefix, rout, c.opt_vprefix,
                              c.loss_vprefix, c.lossd_vprefix, c.lr_vprefix);
  m->nn_reward = nn_make_mlp(L, c.hidden_reward, rout, c.opt_reward,
                             c.loss_reward, c.lossd_reward, c.lr_reward);

  if (!m->nn_repr || !m->nn_dyn || !m->nn_pred || !m->nn_vprefix ||
      !m->nn_reward) {
    mu_model_free(m);
    return NULL;
  }

  m->runtime = mu_runtime_create(m, 0.95f);
  m->repr = mu_model_repr_nn;
  m->predict = mu_model_predict_nn;
  m->dynamics = mu_model_dynamics_nn;
  m->train_policy_value = mu_model_train_policy_value_nn;
  m->train_dynamics = mu_model_train_dynamics_nn;
  m->train_unroll = mu_model_train_unroll_nn;
  m->use_nn = 1;

  return m;
}

MuModel *mu_model_create_nn(const MuConfig *cfg) {
  return mu_model_create_nn_with_cfg(cfg, NULL);
}

/* ------------------------
   Free model
   ------------------------ */
void mu_model_free(MuModel *m) {
  if (!m)
    return;
  if (m->runtime)
    mu_runtime_free((MuRuntime *)m->runtime);
  if (m->nn_repr)
    NN_destroy(m->nn_repr);
  if (m->nn_dyn)
    NN_destroy(m->nn_dyn);
  if (m->nn_pred)
    NN_destroy(m->nn_pred);
  if (m->nn_vprefix)
    NN_destroy(m->nn_vprefix);
  if (m->nn_reward)
    NN_destroy(m->nn_reward);
  free(m->action_embed);
  free(m->repr_W);
  free(m->dyn_W);
  free(m->pred_W);
  free(m->rew_W);
  free(m->vprefix_W);
  free(m);
}

void mu_model_copy_weights(MuModel *dst, const MuModel *src) {
  if (!dst || !src)
    return;

  dst->value_norm_enabled = src->value_norm_enabled;
  dst->value_min = src->value_min;
  dst->value_max = src->value_max;
  dst->value_rescale_eps = src->value_rescale_eps;

  dst->use_value_support = src->use_value_support;
  dst->use_reward_support = src->use_reward_support;
  dst->support_size = src->support_size;
  dst->support_min = src->support_min;
  dst->support_max = src->support_max;
  dst->action_embed_dim = src->action_embed_dim;

  dst->w_policy = src->w_policy;
  dst->w_value = src->w_value;
  dst->w_vprefix = src->w_vprefix;
  dst->w_latent = src->w_latent;
  dst->w_reward = src->w_reward;
  dst->grad_clip = src->grad_clip;

  if (dst->use_nn && src->use_nn) {
    nn_copy_weights(dst->nn_repr, src->nn_repr);
    nn_copy_weights(dst->nn_dyn, src->nn_dyn);
    nn_copy_weights(dst->nn_pred, src->nn_pred);
    nn_copy_weights(dst->nn_vprefix, src->nn_vprefix);
    nn_copy_weights(dst->nn_reward, src->nn_reward);
    if (dst->action_embed && src->action_embed &&
        dst->action_embed_count == src->action_embed_count) {
      memcpy(dst->action_embed, src->action_embed,
             sizeof(float) * (size_t)src->action_embed_count);
    }
    return;
  }

  if (dst->repr_W && src->repr_W && dst->repr_W_count == src->repr_W_count)
    memcpy(dst->repr_W, src->repr_W,
           sizeof(float) * (size_t)src->repr_W_count);
  if (dst->dyn_W && src->dyn_W && dst->dyn_W_count == src->dyn_W_count)
    memcpy(dst->dyn_W, src->dyn_W,
           sizeof(float) * (size_t)src->dyn_W_count);
  if (dst->pred_W && src->pred_W && dst->pred_W_count == src->pred_W_count)
    memcpy(dst->pred_W, src->pred_W,
           sizeof(float) * (size_t)src->pred_W_count);
  if (dst->rew_W && src->rew_W && dst->rew_W_count == src->rew_W_count)
    memcpy(dst->rew_W, src->rew_W,
           sizeof(float) * (size_t)src->rew_W_count);
  if (dst->vprefix_W && src->vprefix_W &&
      dst->vprefix_W_count == src->vprefix_W_count)
    memcpy(dst->vprefix_W, src->vprefix_W,
           sizeof(float) * (size_t)src->vprefix_W_count);
  dst->rew_b = src->rew_b;
  dst->vprefix_b = src->vprefix_b;
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

static float cross_entropy_logits(const float *pi_target, const float *logits,
                                  int A) {
  float m = logits[0];
  for (int i = 1; i < A; i++)
    if (logits[i] > m)
      m = logits[i];

  float sum = 0.0f;
  for (int i = 0; i < A; i++)
    sum += expf(logits[i] - m);
  float logZ = logf(sum) + m;

  float loss = 0.0f;
  for (int i = 0; i < A; i++)
    loss -= pi_target[i] * (logits[i] - logZ);
  return loss;
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
                              const float *weights, int B, float lr) {
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
    value_range_update(m, z_t);
    float z_norm = value_normalize(m, z_t);

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

    float w = weights ? weights[i] : 1.0f;

    // policy gradient: dL/dlogits = (p - pi_target)
    for (int a = 0; a < A; a++)
      dlogits[a] = w * (probs[a] - pi_t[a]);

    // value gradient (MSE): v = tanh(v_lin)
    // d/dv = 2*(v - z); dv/dv_lin = (1 - v^2)
    float dv_lin = w * 2.0f * (v - z_norm) * (1.0f - v * v);

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
                                       const float *next_obs_batch,
                                       const float *weights,
                                       int train_reward_head, int B, float lr,
                                       float *out_latent_mse,
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
    float w = weights ? weights[i] : 1.0f;

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

    float r_pred = 0.0f;
    if (train_reward_head) {
      float r_lin = m->rew_b;
      for (int li = 0; li < L; li++)
        r_lin += m->rew_W[li] * h2[li];
      r_pred = tanhf(r_lin);
    }

    // accumulate losses (for logging)
    // latent mse
    for (int li = 0; li < L; li++) {
      float d = h2[li] - h_tgt[li];
      latent_mse_acc += w * d * d;
    }
    // reward mse
    if (train_reward_head) {
      float dr = (r_pred - r_tgt);
      reward_mse_acc += w * dr * dr;
    }

    // grads: reward mse through tanh
    float drlin = 0.0f;
    if (train_reward_head) {
      drlin = w * 2.0f * (r_pred - r_tgt) * (1.0f - r_pred * r_pred);

      g_rewB += drlin;
      for (int li = 0; li < L; li++)
        g_rewW[li] += drlin * h2[li];
    }

    // backprop into dyn via h2
    for (int li = 0; li < L; li++) {
      float d_h2 = w * 2.0f * (h2[li] - h_tgt[li]) +
                   drlin * (train_reward_head ? m->rew_W[li] : 0.0f);
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
  if (train_reward_head) {
    for (int li = 0; li < L; li++)
      m->rew_W[li] -= scale * g_rewW[li];
    m->rew_b -= scale * g_rewB;
  }

  free(h);
  free(h2);
  free(h_tgt);
  free(pre2);
  free(g_dyn);
  free(g_rewW);
}

void muzero_model_train_unroll_batch(
    MuModel *m, const float *obs_seq, const float *pi_seq, const float *z_seq,
    const float *vprefix_seq, const int *a_seq, const float *r_seq,
    const int *done_seq, int B, int unroll_steps, int bootstrap_steps,
    float discount, float lr, const float *weights, float *out_policy_loss,
    float *out_value_loss, float *out_reward_loss, float *out_latent_loss) {
  if (out_policy_loss)
    *out_policy_loss = 0.0f;
  if (out_value_loss)
    *out_value_loss = 0.0f;
  if (out_reward_loss)
    *out_reward_loss = 0.0f;
  if (out_latent_loss)
    *out_latent_loss = 0.0f;

  if (!m || !obs_seq || !pi_seq || !z_seq || !a_seq || !r_seq || !done_seq)
    return;
  if (B <= 0 || unroll_steps < 0 || !(lr > 0.0f))
    return;
  if (!(discount > 0.0f))
    discount = 0.997f;
  if (bootstrap_steps <= 0)
    bootstrap_steps = 1;

  const int O = m->cfg.obs_dim;
  const int L = m->cfg.latent_dim;
  const int A = m->cfg.action_count;
  const int K = unroll_steps;

  size_t steps = (size_t)K + 1;

  float *g_repr = (float *)calloc((size_t)L * (size_t)O, sizeof(float));
  float *g_pred = (float *)calloc((size_t)(A + 1) * (size_t)L, sizeof(float));
  float *g_dyn = (float *)calloc((size_t)L * (size_t)(L + 1), sizeof(float));
  float *g_rewW = (float *)calloc((size_t)L, sizeof(float));
  float g_rewB = 0.0f;

  float *h = (float *)malloc(sizeof(float) * steps * (size_t)L);
  float *h_tgt = (float *)malloc(sizeof(float) * steps * (size_t)L);
  float *a_norm = (float *)malloc(sizeof(float) * (size_t)K);
  float *logits = (float *)malloc(sizeof(float) * steps * (size_t)A);
  float *probs = (float *)malloc(sizeof(float) * steps * (size_t)A);
  float *values = (float *)malloc(sizeof(float) * steps);
  float *rewards = (float *)malloc(sizeof(float) * (size_t)K);
  float *d_h = (float *)malloc(sizeof(float) * steps * (size_t)L);
  float *mask = (float *)malloc(sizeof(float) * steps);

  if (!g_repr || !g_pred || !g_dyn || !g_rewW || !h || !h_tgt || !a_norm ||
      !logits || !probs || !values || !rewards || !d_h || !mask) {
    free(g_repr);
    free(g_pred);
    free(g_dyn);
    free(g_rewW);
    free(h);
    free(h_tgt);
    free(a_norm);
    free(logits);
    free(probs);
    free(values);
    free(rewards);
    free(d_h);
    free(mask);
    return;
  }

  double policy_loss_acc = 0.0;
  double value_loss_acc = 0.0;
  double reward_loss_acc = 0.0;
  double latent_loss_acc = 0.0;
  double policy_cnt = 0.0;
  double value_cnt = 0.0;
  double reward_cnt = 0.0;
  double latent_cnt = 0.0;

  for (int b = 0; b < B; b++) {
    const float *obs0 = obs_seq + (size_t)b * steps * (size_t)O;
    const float *pi0 = pi_seq + (size_t)b * steps * (size_t)A;
    const float *z0 = z_seq + (size_t)b * steps;
    const float *vp0 = vprefix_seq ? (vprefix_seq + (size_t)b * steps) : NULL;
    const int *a0 = a_seq + (size_t)b * (size_t)K;
    const float *r0 = r_seq + (size_t)b * (size_t)K;
    const int *d0 = done_seq + (size_t)b * (size_t)K;

    mask[0] = 1.0f;
    for (int k = 0; k < K; k++) {
      mask[k + 1] = (mask[k] > 0.0f && d0[k] == 0) ? 1.0f : 0.0f;
    }

    // h0 = repr(s0)
    mu_model_repr(m, obs0, h);

    // rollout dynamics + reward, and compute target latents
    for (int k = 0; k < K; k++) {
      int act = a0[k];
      float a =
          (A > 1) ? (float)act / (float)(A - 1) : 0.0f; // [0,1]
      a_norm[k] = a * 2.0f - 1.0f;                      // [-1,1]

      float *h_k = h + (size_t)k * (size_t)L;
      float *h_k1 = h + (size_t)(k + 1) * (size_t)L;
      for (int li = 0; li < L; li++) {
        float sum = 0.0f;
        const float *row = &m->dyn_W[li * (L + 1)];
        for (int j = 0; j < L; j++)
          sum += h_k[j] * row[j];
        sum += a_norm[k] * row[L];
        h_k1[li] = tanhf(sum);
      }

      // target latent from repr(s_{k+1}) with stop-grad
      const float *obs_k1 = obs0 + (size_t)(k + 1) * (size_t)O;
      mu_model_repr(m, obs_k1, h_tgt + (size_t)(k + 1) * (size_t)L);

      // value prefix prediction from h_{k+1}
      float vp_lin = m->vprefix_b;
      for (int li = 0; li < L; li++)
        vp_lin += m->vprefix_W[li] * h_k1[li];
      rewards[k] = tanhf(vp_lin);
    }

    // policy/value prediction for all k
    for (int k = 0; k < K + 1; k++) {
      float *h_k = h + (size_t)k * (size_t)L;
      float *logits_k = logits + (size_t)k * (size_t)A;

      for (int a = 0; a < A; a++) {
        float sum = 0.0f;
        for (int j = 0; j < L; j++)
          sum += h_k[j] * m->pred_W[a * L + j];
        logits_k[a] = sum;
      }

      float v_lin = 0.0f;
      int value_off = A * L;
      for (int j = 0; j < L; j++)
        v_lin += h_k[j] * m->pred_W[value_off + j];
      values[k] = tanhf(v_lin);
    }

    memset(d_h, 0, sizeof(float) * steps * (size_t)L);

    float w_sample = weights ? weights[b] : 1.0f;
    for (int k = K; k >= 0; k--) {
      float *h_k = h + (size_t)k * (size_t)L;
      float *d_h_k = d_h + (size_t)k * (size_t)L;
      const float *pi_k = pi0 + (size_t)k * (size_t)A;
      float z_k = z0[k];

      if (mask[k] > 0.0f) {
        float *logits_k = logits + (size_t)k * (size_t)A;
        float *probs_k = probs + (size_t)k * (size_t)A;
        softmaxf(logits_k, A, probs_k);

        policy_loss_acc += w_sample * cross_entropy_logits(pi_k, logits_k, A);
        policy_cnt += 1.0;

        for (int a = 0; a < A; a++) {
          float dlog = w_sample * (probs_k[a] - pi_k[a]);
          for (int j = 0; j < L; j++) {
            g_pred[a * L + j] += dlog * h_k[j];
            d_h_k[j] += dlog * m->pred_W[a * L + j];
          }
        }

        float v = values[k];

        float target_v = z_k;
        if (bootstrap_steps > 0) {
          float G = 0.0f;
          float gamma_pow = 1.0f;
          int done_flag = 0;
          int max_i = bootstrap_steps;
          for (int i = 0; i < max_i; i++) {
            int t = k + i;
            if (t >= K)
              break;
            G += gamma_pow * r0[t];
            gamma_pow *= discount;
            if (done_seq && d0[t]) {
              done_flag = 1;
              break;
            }
          }

          float bootstrap = 0.0f;
          if (!done_flag) {
            int t_boot = k + max_i;
            if (t_boot > K)
              t_boot = K;
            bootstrap = mu_model_denorm_value(m, values[t_boot]);
          }

          target_v = G + gamma_pow * bootstrap;
        }
        value_range_update(m, target_v);
        float target_norm = value_normalize(m, target_v);

        float dv_lin =
            w_sample * 2.0f * (v - target_norm) * (1.0f - v * v);
        value_loss_acc +=
            w_sample * (double)(v - target_norm) * (double)(v - target_norm);
        value_cnt += 1.0;

        int value_off = A * L;
        for (int j = 0; j < L; j++) {
          g_pred[value_off + j] += dv_lin * h_k[j];
          d_h_k[j] += dv_lin * m->pred_W[value_off + j];
        }
      }

      if (k < K && mask[k] > 0.0f) {
        float *h_k1 = h + (size_t)(k + 1) * (size_t)L;
        float *d_h_k1 = d_h + (size_t)(k + 1) * (size_t)L;

        float vp_pred = rewards[k];
        float vp_tgt = r0[k];
        if (vp0) {
          vp_tgt = vp0[k];
        } else {
          float prefix = 0.0f;
          float gamma_pow = 1.0f;
          for (int i = 0; i <= k; i++) {
            prefix += gamma_pow * r0[i];
            gamma_pow *= discount;
          }
          vp_tgt = prefix;
        }
        float dvp_lin =
            w_sample * 2.0f * (vp_pred - vp_tgt) * (1.0f - vp_pred * vp_pred);

        reward_loss_acc +=
            w_sample * (double)(vp_pred - vp_tgt) * (double)(vp_pred - vp_tgt);
        reward_cnt += 1.0;

        g_rewB += dvp_lin;
        for (int j = 0; j < L; j++) {
          g_rewW[j] += dvp_lin * h_k1[j];
          d_h_k1[j] += dvp_lin * m->vprefix_W[j];
        }

        if (mask[k + 1] > 0.0f) {
          float *h_t = h_tgt + (size_t)(k + 1) * (size_t)L;
          for (int j = 0; j < L; j++) {
            float d = h_k1[j] - h_t[j];
            latent_loss_acc += w_sample * (double)d * (double)d;
            latent_cnt += 1.0;
            d_h_k1[j] += w_sample * 2.0f * d;
          }
        }

        // backprop through dynamics
        float *d_h_k_prev = d_h + (size_t)k * (size_t)L;

        for (int li = 0; li < L; li++) {
          float hval = h_k1[li];
          float dpre = d_h_k1[li] * (1.0f - hval * hval);

          float *g_row = &g_dyn[li * (L + 1)];
          const float *row = &m->dyn_W[li * (L + 1)];

          for (int j = 0; j < L; j++) {
            g_row[j] += dpre * h_k[j];
            d_h_k_prev[j] += dpre * row[j];
          }
          g_row[L] += dpre * a_norm[k];
        }
      }
    }

    // backprop through representation for h0
    if (mask[0] > 0.0f) {
      for (int li = 0; li < L; li++) {
        float hval = h[li];
        float dpre = d_h[li] * (1.0f - hval * hval);
        for (int j = 0; j < O; j++)
          g_repr[li * O + j] += dpre * obs0[j];
      }
    }
  }

  float scale = lr / (float)(B * (K + 1));

  for (int idx = 0; idx < L * O; idx++)
    m->repr_W[idx] -= scale * g_repr[idx];
  for (int idx = 0; idx < (A + 1) * L; idx++)
    m->pred_W[idx] -= scale * g_pred[idx];
  for (int idx = 0; idx < L * (L + 1); idx++)
    m->dyn_W[idx] -= scale * g_dyn[idx];
  for (int j = 0; j < L; j++)
    m->vprefix_W[j] -= scale * g_rewW[j];
  m->vprefix_b -= scale * g_rewB;

  if (out_policy_loss && policy_cnt > 0.0)
    *out_policy_loss = (float)(policy_loss_acc / policy_cnt);
  if (out_value_loss && value_cnt > 0.0)
    *out_value_loss = (float)(value_loss_acc / value_cnt);
  if (out_reward_loss && reward_cnt > 0.0)
    *out_reward_loss = (float)(reward_loss_acc / reward_cnt);
  if (out_latent_loss && latent_cnt > 0.0)
    *out_latent_loss = (float)(latent_loss_acc / latent_cnt);

  free(g_repr);
  free(g_pred);
  free(g_dyn);
  free(g_rewW);
  free(h);
  free(h_tgt);
  free(a_norm);
  free(logits);
  free(probs);
  free(values);
  free(rewards);
  free(d_h);
  free(mask);
}

enum {
  MUZ_MAGIC = 0x4d5a5631u,
  MUZ_VERSION = 1u,
  MUZ_MAGIC_NN = 0x4d5a4e31u,
  MUZ_VERSION_NN = 2u
};

int mu_model_save(MuModel *m, const char *filename) {
  if (!m || !filename)
    return 0;

  if (m->use_nn) {
    FILE *f = fopen(filename, "wb");
    if (!f)
      return 0;

    uint32_t magic = MUZ_MAGIC_NN;
    uint32_t version = MUZ_VERSION_NN;
    if (fwrite(&magic, sizeof(magic), 1, f) != 1 ||
        fwrite(&version, sizeof(version), 1, f) != 1) {
      fclose(f);
      return 0;
    }

    if (fwrite(&m->cfg, sizeof(MuConfig), 1, f) != 1 ||
        fwrite(&m->value_norm_enabled, sizeof(int), 1, f) != 1 ||
        fwrite(&m->value_min, sizeof(float), 1, f) != 1 ||
        fwrite(&m->value_max, sizeof(float), 1, f) != 1 ||
        fwrite(&m->value_rescale_eps, sizeof(float), 1, f) != 1 ||
        fwrite(&m->use_value_support, sizeof(int), 1, f) != 1 ||
        fwrite(&m->use_reward_support, sizeof(int), 1, f) != 1 ||
        fwrite(&m->support_size, sizeof(int), 1, f) != 1 ||
        fwrite(&m->support_min, sizeof(float), 1, f) != 1 ||
        fwrite(&m->support_max, sizeof(float), 1, f) != 1 ||
        fwrite(&m->action_embed_dim, sizeof(int), 1, f) != 1 ||
        fwrite(&m->action_embed_count, sizeof(int), 1, f) != 1 ||
        fwrite(&m->w_policy, sizeof(float), 1, f) != 1 ||
        fwrite(&m->w_value, sizeof(float), 1, f) != 1 ||
        fwrite(&m->w_vprefix, sizeof(float), 1, f) != 1 ||
        fwrite(&m->w_latent, sizeof(float), 1, f) != 1 ||
        fwrite(&m->w_reward, sizeof(float), 1, f) != 1 ||
        fwrite(&m->grad_clip, sizeof(float), 1, f) != 1) {
      fclose(f);
      return 0;
    }

    uint32_t mask = 0;
    if (m->nn_repr)
      mask |= 1u << 0;
    if (m->nn_dyn)
      mask |= 1u << 1;
    if (m->nn_pred)
      mask |= 1u << 2;
    if (m->nn_vprefix)
      mask |= 1u << 3;
    if (m->nn_reward)
      mask |= 1u << 4;
    if (fwrite(&mask, sizeof(mask), 1, f) != 1) {
      fclose(f);
      return 0;
    }

    if (m->action_embed && m->action_embed_count > 0) {
      if (fwrite(m->action_embed, sizeof(float),
                 (size_t)m->action_embed_count, f) !=
          (size_t)m->action_embed_count) {
        fclose(f);
        return 0;
      }
    }

    fclose(f);

    char path[512];
    if (m->nn_repr) {
      snprintf(path, sizeof(path), "%s.repr.nn", filename);
      if (!NN_save(m->nn_repr, path))
        return 0;
    }
    if (m->nn_dyn) {
      snprintf(path, sizeof(path), "%s.dyn.nn", filename);
      if (!NN_save(m->nn_dyn, path))
        return 0;
    }
    if (m->nn_pred) {
      snprintf(path, sizeof(path), "%s.pred.nn", filename);
      if (!NN_save(m->nn_pred, path))
        return 0;
    }
    if (m->nn_vprefix) {
      snprintf(path, sizeof(path), "%s.vprefix.nn", filename);
      if (!NN_save(m->nn_vprefix, path))
        return 0;
    }
    if (m->nn_reward) {
      snprintf(path, sizeof(path), "%s.reward.nn", filename);
      if (!NN_save(m->nn_reward, path))
        return 0;
    }

    return 1;
  }

  FILE *f = fopen(filename, "wb");
  if (!f)
    return 0;

  uint32_t magic = MUZ_MAGIC;
  uint32_t version = MUZ_VERSION;
  if (fwrite(&magic, sizeof(magic), 1, f) != 1 ||
      fwrite(&version, sizeof(version), 1, f) != 1) {
    fclose(f);
    return 0;
  }

  if (fwrite(&m->cfg, sizeof(MuConfig), 1, f) != 1) {
    fclose(f);
    return 0;
  }

  if (fwrite(&m->value_norm_enabled, sizeof(int), 1, f) != 1 ||
      fwrite(&m->value_min, sizeof(float), 1, f) != 1 ||
      fwrite(&m->value_max, sizeof(float), 1, f) != 1 ||
      fwrite(&m->value_rescale_eps, sizeof(float), 1, f) != 1) {
    fclose(f);
    return 0;
  }

  if (fwrite(&m->repr_W_count, sizeof(int), 1, f) != 1 ||
      fwrite(&m->dyn_W_count, sizeof(int), 1, f) != 1 ||
      fwrite(&m->pred_W_count, sizeof(int), 1, f) != 1 ||
      fwrite(&m->rew_W_count, sizeof(int), 1, f) != 1 ||
      fwrite(&m->vprefix_W_count, sizeof(int), 1, f) != 1 ||
      fwrite(&m->rew_b, sizeof(float), 1, f) != 1 ||
      fwrite(&m->vprefix_b, sizeof(float), 1, f) != 1) {
    fclose(f);
    return 0;
  }

  if (m->repr_W_count > 0 &&
      fwrite(m->repr_W, sizeof(float), (size_t)m->repr_W_count, f) !=
          (size_t)m->repr_W_count) {
    fclose(f);
    return 0;
  }
  if (m->dyn_W_count > 0 &&
      fwrite(m->dyn_W, sizeof(float), (size_t)m->dyn_W_count, f) !=
          (size_t)m->dyn_W_count) {
    fclose(f);
    return 0;
  }
  if (m->pred_W_count > 0 &&
      fwrite(m->pred_W, sizeof(float), (size_t)m->pred_W_count, f) !=
          (size_t)m->pred_W_count) {
    fclose(f);
    return 0;
  }
  if (m->rew_W_count > 0 &&
      fwrite(m->rew_W, sizeof(float), (size_t)m->rew_W_count, f) !=
          (size_t)m->rew_W_count) {
    fclose(f);
    return 0;
  }
  if (m->vprefix_W_count > 0 &&
      fwrite(m->vprefix_W, sizeof(float), (size_t)m->vprefix_W_count, f) !=
          (size_t)m->vprefix_W_count) {
    fclose(f);
    return 0;
  }

  fclose(f);
  return 1;
}

MuModel *mu_model_load(const char *filename) {
  if (!filename)
    return NULL;

  FILE *f = fopen(filename, "rb");
  if (!f)
    return NULL;

  uint32_t magic = 0;
  uint32_t version = 0;
  if (fread(&magic, sizeof(magic), 1, f) != 1 ||
      fread(&version, sizeof(version), 1, f) != 1) {
    fclose(f);
    return NULL;
  }
  if (magic == MUZ_MAGIC_NN && version == MUZ_VERSION_NN) {
    MuConfig cfg;
    int value_norm_enabled = 0;
    float value_min = -1.0f;
    float value_max = 1.0f;
    float value_rescale_eps = 0.001f;
    int use_value_support = 0;
    int use_reward_support = 0;
    int support_size = 1;
    float support_min = -2.0f;
    float support_max = 2.0f;
    int action_embed_dim = 0;
    int action_embed_count = 0;
    float w_policy = 1.0f;
    float w_value = 1.0f;
    float w_vprefix = 1.0f;
    float w_latent = 1.0f;
    float w_reward = 1.0f;
    float grad_clip = 0.0f;
    uint32_t mask = 0;

    if (fread(&cfg, sizeof(MuConfig), 1, f) != 1 ||
        fread(&value_norm_enabled, sizeof(int), 1, f) != 1 ||
        fread(&value_min, sizeof(float), 1, f) != 1 ||
        fread(&value_max, sizeof(float), 1, f) != 1 ||
        fread(&value_rescale_eps, sizeof(float), 1, f) != 1 ||
        fread(&use_value_support, sizeof(int), 1, f) != 1 ||
        fread(&use_reward_support, sizeof(int), 1, f) != 1 ||
        fread(&support_size, sizeof(int), 1, f) != 1 ||
        fread(&support_min, sizeof(float), 1, f) != 1 ||
        fread(&support_max, sizeof(float), 1, f) != 1 ||
        fread(&action_embed_dim, sizeof(int), 1, f) != 1 ||
        fread(&action_embed_count, sizeof(int), 1, f) != 1 ||
        fread(&w_policy, sizeof(float), 1, f) != 1 ||
        fread(&w_value, sizeof(float), 1, f) != 1 ||
        fread(&w_vprefix, sizeof(float), 1, f) != 1 ||
        fread(&w_latent, sizeof(float), 1, f) != 1 ||
        fread(&w_reward, sizeof(float), 1, f) != 1 ||
        fread(&grad_clip, sizeof(float), 1, f) != 1 ||
        fread(&mask, sizeof(mask), 1, f) != 1) {
      fclose(f);
      return NULL;
    }
    float *action_embed = NULL;
    if (action_embed_count > 0) {
      action_embed =
          (float *)malloc(sizeof(float) * (size_t)action_embed_count);
      if (!action_embed ||
          fread(action_embed, sizeof(float), (size_t)action_embed_count, f) !=
              (size_t)action_embed_count) {
        free(action_embed);
        fclose(f);
        return NULL;
      }
    }
    fclose(f);

    MuNNConfig nn_cfg = mu_nn_default_cfg(&cfg);
    nn_cfg.use_value_support = use_value_support;
    nn_cfg.use_reward_support = use_reward_support;
    nn_cfg.support_size = support_size;
    nn_cfg.support_min = support_min;
    nn_cfg.support_max = support_max;
    nn_cfg.action_embed_dim = action_embed_dim;
    nn_cfg.w_policy = w_policy;
    nn_cfg.w_value = w_value;
    nn_cfg.w_vprefix = w_vprefix;
    nn_cfg.w_latent = w_latent;
    nn_cfg.w_reward = w_reward;
    nn_cfg.grad_clip = grad_clip;

    MuModel *m = mu_model_create_nn_with_cfg(&cfg, &nn_cfg);
    if (!m)
      return NULL;

    m->value_norm_enabled = value_norm_enabled;
    m->value_min = value_min;
    m->value_max = value_max;
    m->value_rescale_eps = value_rescale_eps;
    if (action_embed && m->action_embed &&
        m->action_embed_count == action_embed_count) {
      memcpy(m->action_embed, action_embed,
             sizeof(float) * (size_t)action_embed_count);
    }
    free(action_embed);

    char path[512];
    if (mask & (1u << 0)) {
      snprintf(path, sizeof(path), "%s.repr.nn", filename);
      NN_destroy(m->nn_repr);
      m->nn_repr = NN_load(path);
    }
    if (mask & (1u << 1)) {
      snprintf(path, sizeof(path), "%s.dyn.nn", filename);
      NN_destroy(m->nn_dyn);
      m->nn_dyn = NN_load(path);
    }
    if (mask & (1u << 2)) {
      snprintf(path, sizeof(path), "%s.pred.nn", filename);
      NN_destroy(m->nn_pred);
      m->nn_pred = NN_load(path);
    }
    if (mask & (1u << 3)) {
      snprintf(path, sizeof(path), "%s.vprefix.nn", filename);
      NN_destroy(m->nn_vprefix);
      m->nn_vprefix = NN_load(path);
    }
    if (mask & (1u << 4)) {
      snprintf(path, sizeof(path), "%s.reward.nn", filename);
      NN_destroy(m->nn_reward);
      m->nn_reward = NN_load(path);
    }

    if ((mask & (1u << 0)) && !m->nn_repr) {
      mu_model_free(m);
      return NULL;
    }
    if ((mask & (1u << 1)) && !m->nn_dyn) {
      mu_model_free(m);
      return NULL;
    }
    if ((mask & (1u << 2)) && !m->nn_pred) {
      mu_model_free(m);
      return NULL;
    }
    if ((mask & (1u << 3)) && !m->nn_vprefix) {
      mu_model_free(m);
      return NULL;
    }
    if ((mask & (1u << 4)) && !m->nn_reward) {
      mu_model_free(m);
      return NULL;
    }

    return m;
  }

  if (magic != MUZ_MAGIC || version != MUZ_VERSION) {
    fclose(f);
    return NULL;
  }

  MuConfig cfg;
  if (fread(&cfg, sizeof(MuConfig), 1, f) != 1) {
    fclose(f);
    return NULL;
  }

  int value_norm_enabled = 0;
  float value_min = -1.0f;
  float value_max = 1.0f;
  float value_rescale_eps = 0.001f;
  if (fread(&value_norm_enabled, sizeof(int), 1, f) != 1 ||
      fread(&value_min, sizeof(float), 1, f) != 1 ||
      fread(&value_max, sizeof(float), 1, f) != 1 ||
      fread(&value_rescale_eps, sizeof(float), 1, f) != 1) {
    fclose(f);
    return NULL;
  }

  int repr_cnt = 0;
  int dyn_cnt = 0;
  int pred_cnt = 0;
  int rew_cnt = 0;
  int vprefix_cnt = 0;
  float rew_b = 0.0f;
  float vprefix_b = 0.0f;
  if (fread(&repr_cnt, sizeof(int), 1, f) != 1 ||
      fread(&dyn_cnt, sizeof(int), 1, f) != 1 ||
      fread(&pred_cnt, sizeof(int), 1, f) != 1 ||
      fread(&rew_cnt, sizeof(int), 1, f) != 1 ||
      fread(&vprefix_cnt, sizeof(int), 1, f) != 1 ||
      fread(&rew_b, sizeof(float), 1, f) != 1 ||
      fread(&vprefix_b, sizeof(float), 1, f) != 1) {
    fclose(f);
    return NULL;
  }

  int exp_repr = cfg.obs_dim * cfg.latent_dim;
  int exp_dyn = cfg.latent_dim * (cfg.latent_dim + 1);
  int exp_pred = cfg.latent_dim * (cfg.action_count + 1);
  int exp_rew = cfg.latent_dim;
  int exp_vprefix = cfg.latent_dim;
  if (repr_cnt != exp_repr || dyn_cnt != exp_dyn || pred_cnt != exp_pred ||
      rew_cnt != exp_rew || vprefix_cnt != exp_vprefix) {
    fclose(f);
    return NULL;
  }

  MuModel *m = (MuModel *)calloc(1, sizeof(MuModel));
  if (!m) {
    fclose(f);
    return NULL;
  }

  m->cfg = cfg;
  m->repr_W_count = repr_cnt;
  m->dyn_W_count = dyn_cnt;
  m->pred_W_count = pred_cnt;
  m->rew_W_count = rew_cnt;
  m->vprefix_W_count = vprefix_cnt;
  m->rew_b = rew_b;
  m->vprefix_b = vprefix_b;

  m->repr_W = (float *)malloc(sizeof(float) * (size_t)repr_cnt);
  m->dyn_W = (float *)malloc(sizeof(float) * (size_t)dyn_cnt);
  m->pred_W = (float *)malloc(sizeof(float) * (size_t)pred_cnt);
  m->rew_W = (float *)malloc(sizeof(float) * (size_t)rew_cnt);
  m->vprefix_W = (float *)malloc(sizeof(float) * (size_t)vprefix_cnt);

  if (!m->repr_W || !m->dyn_W || !m->pred_W || !m->rew_W || !m->vprefix_W) {
    mu_model_free(m);
    fclose(f);
    return NULL;
  }

  if (fread(m->repr_W, sizeof(float), (size_t)repr_cnt, f) !=
          (size_t)repr_cnt ||
      fread(m->dyn_W, sizeof(float), (size_t)dyn_cnt, f) !=
          (size_t)dyn_cnt ||
      fread(m->pred_W, sizeof(float), (size_t)pred_cnt, f) !=
          (size_t)pred_cnt ||
      fread(m->rew_W, sizeof(float), (size_t)rew_cnt, f) !=
          (size_t)rew_cnt ||
      fread(m->vprefix_W, sizeof(float), (size_t)vprefix_cnt, f) !=
          (size_t)vprefix_cnt) {
    mu_model_free(m);
    fclose(f);
    return NULL;
  }

  m->value_norm_enabled = value_norm_enabled;
  m->value_min = value_min;
  m->value_max = value_max;
  m->value_rescale_eps = value_rescale_eps;

  m->runtime = mu_runtime_create(m, 0.95f);
  m->repr = NULL;
  m->dynamics = NULL;
  m->predict = NULL;
  m->train_policy_value = NULL;
  m->train_dynamics = NULL;
  m->train_unroll = NULL;

  fclose(f);
  return m;
}
