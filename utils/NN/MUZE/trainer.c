#include "trainer.h"
#include "muzero_model.h"
#include "replay_buffer.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

static float mse(float a, float b) {
  float d = a - b;
  return d * d;
}

static float cross_entropy_logits(const float *pi_target, const float *logits,
                                  int A) {
  // find max logit for stability
  float m = logits[0];
  for (int i = 1; i < A; i++)
    if (logits[i] > m)
      m = logits[i];

  // logsumexp
  float sum = 0.0f;
  for (int i = 0; i < A; i++)
    sum += expf(logits[i] - m);
  const float logZ = logf(sum) + m;

  float loss = 0.0f;
  for (int i = 0; i < A; i++) {
    // log_softmax = logits - logZ
    loss -= pi_target[i] * (logits[i] - logZ);
  }
  return loss;
}

static bool is_finite_float(float x) { return isfinite((double)x); }

void trainer_train_dynamics(MuModel *model, ReplayBuffer *rb,
                            const TrainerConfig *cfg) {
  if (!model || !rb || !cfg)
    return;

  size_t n = rb_size(rb);
  if ((int)n < cfg->min_replay_size)
    return;

  int B = cfg->batch_size;
  if (B <= 0)
    return;

  int steps = cfg->train_steps;
  if (steps <= 0)
    steps = 1;

  float lr = cfg->lr;
  if (!(lr > 0.0f))
    return;

  int O = muzero_model_obs_dim(model);
  if (O <= 0)
    return;

  float *obs_batch = (float *)malloc(sizeof(float) * (size_t)B * (size_t)O);
  float *next_obs_batch =
      (float *)malloc(sizeof(float) * (size_t)B * (size_t)O);
  int *a_batch = (int *)malloc(sizeof(int) * (size_t)B);
  float *r_batch = (float *)malloc(sizeof(float) * (size_t)B);
  int *done_batch = (int *)malloc(sizeof(int) * (size_t)B);

  if (!obs_batch || !next_obs_batch || !a_batch || !r_batch || !done_batch) {
    free(obs_batch);
    free(next_obs_batch);
    free(a_batch);
    free(r_batch);
    free(done_batch);
    return;
  }

  for (int t = 0; t < steps; t++) {
    int actual = rb_sample_transition(rb, B, obs_batch, a_batch, r_batch,
                                      next_obs_batch, done_batch);
    if (actual <= 0)
      break;

    float lat_mse = 0.0f;
    float rew_mse = 0.0f;

    if (model->train_dynamics) {
      model->train_dynamics(model, obs_batch, a_batch, r_batch, next_obs_batch,
                            actual, lr, &lat_mse, &rew_mse);
    } else {
      muzero_model_train_dynamics_batch(model, obs_batch, a_batch, r_batch,
                                        next_obs_batch, actual, lr, &lat_mse,
                                        &rew_mse);
    }

    if (t == 0 || (t % 50) == 0) {
      printf("[train dyn] step=%d/%d batch=%d latent_mse=%.6f reward_mse=%.6f "
             "replay=%zu\n",
             t + 1, steps, actual, lat_mse, rew_mse, n);
    }
  }

  free(obs_batch);
  free(next_obs_batch);
  free(a_batch);
  free(r_batch);
  free(done_batch);
}

void trainer_train_from_replay(MuModel *model, ReplayBuffer *rb,
                               const TrainerConfig *cfg) {
  if (!model || !rb || !cfg)
    return;

  size_t n = rb_size(rb);
  if ((int)n < cfg->min_replay_size)
    return;

  int B = cfg->batch_size;
  if (B <= 0)
    return;

  int steps = cfg->train_steps;
  if (steps <= 0)
    steps = 1;

  float lr = cfg->lr;
  if (!(lr > 0.0f))
    return;
  int K = cfg->unroll_steps;
  int n_boot = cfg->bootstrap_steps;
  float discount = cfg->discount;
  if (!(discount > 0.0f))
    discount = 0.997f;
  int use_per = cfg->use_per;
  float per_alpha = cfg->per_alpha;
  float per_beta = cfg->per_beta;
  float per_eps = cfg->per_eps;
  if (per_alpha <= 0.0f)
    per_alpha = 0.6f;
  if (per_beta <= 0.0f)
    per_beta = 0.4f;
  if (per_eps <= 0.0f)
    per_eps = 1e-3f;

  int O = muzero_model_obs_dim(model);
  int A = muzero_model_action_count(model);
  int L = model->cfg.latent_dim;
  if (O <= 0 || A <= 0)
    return;

  if (K > 0) {
    size_t steps_per = (size_t)K + 1;

    float *obs_seq =
        (float *)malloc(sizeof(float) * (size_t)B * steps_per * (size_t)O);
    float *pi_seq =
        (float *)malloc(sizeof(float) * (size_t)B * steps_per * (size_t)A);
    float *z_seq =
        (float *)malloc(sizeof(float) * (size_t)B * steps_per);
    float *vprefix_seq =
        (float *)malloc(sizeof(float) * (size_t)B * steps_per);
    int *a_seq = (int *)malloc(sizeof(int) * (size_t)B * (size_t)K);
    float *r_seq = (float *)malloc(sizeof(float) * (size_t)B * (size_t)K);
    int *done_seq = (int *)malloc(sizeof(int) * (size_t)B * (size_t)K);
    size_t *idx_out = (size_t *)malloc(sizeof(size_t) * (size_t)B);
    float *prob_out = (float *)malloc(sizeof(float) * (size_t)B);
    float *w_batch = (float *)malloc(sizeof(float) * (size_t)B);

    if (!obs_seq || !pi_seq || !z_seq || !vprefix_seq || !a_seq || !r_seq ||
        !done_seq || !idx_out || !prob_out || !w_batch) {
      free(obs_seq);
      free(pi_seq);
      free(z_seq);
      free(vprefix_seq);
      free(a_seq);
      free(r_seq);
      free(done_seq);
      free(idx_out);
      free(prob_out);
      free(w_batch);
      return;
    }

    for (int t = 0; t < steps; t++) {
      int actual =
          use_per
              ? rb_sample_sequence_per(rb, B, K, per_alpha, obs_seq, pi_seq,
                                       z_seq, vprefix_seq, a_seq, r_seq,
                                       done_seq, idx_out, prob_out)
              : rb_sample_sequence_vprefix(rb, B, K, obs_seq, pi_seq, z_seq,
                                           vprefix_seq, a_seq, r_seq,
                                           done_seq);
      if (actual <= 0)
        break;

      float w_min = 1.0f;
      float w_max = 1.0f;
      float w_sum = 0.0f;
      if (use_per) {
        const float N = (float)rb->size;
        float max_w = 1.0f;
        for (int i = 0; i < actual; i++) {
          float p = prob_out[i];
          float w = powf(N * p, -per_beta);
          w_batch[i] = w;
          if (w > max_w)
            max_w = w;
        }
        if (max_w > 0.0f) {
          float inv = 1.0f / max_w;
          for (int i = 0; i < actual; i++)
            w_batch[i] *= inv;
        }
        w_min = w_batch[0];
        w_max = w_batch[0];
        for (int i = 0; i < actual; i++) {
          float w = w_batch[i];
          if (w < w_min)
            w_min = w;
          if (w > w_max)
            w_max = w;
          w_sum += w;
        }
      } else {
        for (int i = 0; i < actual; i++)
          w_batch[i] = 1.0f;
        w_sum = (float)actual;
      }

      float policy_loss = 0.0f;
      float value_loss = 0.0f;
      float reward_loss = 0.0f;
      float latent_loss = 0.0f;

      muzero_model_train_unroll_batch(
          model, obs_seq, pi_seq, z_seq, vprefix_seq, a_seq, r_seq, done_seq,
          actual, K, n_boot, discount, lr, w_batch, &policy_loss, &value_loss,
          &reward_loss, &latent_loss);

      if (use_per && L > 0) {
        float *latent = (float *)malloc(sizeof(float) * (size_t)L);
        float *logits = (float *)malloc(sizeof(float) * (size_t)A);
        if (latent && logits) {
          for (int i = 0; i < actual; i++) {
            const float *obs0 =
                obs_seq + (size_t)i * steps_per * (size_t)O;
            mu_model_repr(model, obs0, latent);
            float v_pred = 0.0f;
            mu_model_predict(model, latent, logits, &v_pred);
            float z_tgt = z_seq[(size_t)i * steps_per];
            float td = fabsf(v_pred - z_tgt);
            rb_set_priority(rb, idx_out[i], td + per_eps);
          }
        }
        free(latent);
        free(logits);
      }

      if (t == 0 || (t % 50) == 0) {
        if (use_per) {
          printf("[train unroll] step=%d/%d batch=%d K=%d pol=%.6f val=%.6f "
                 "rew=%.6f lat=%.6f w[min/mean/max]=%.3f/%.3f/%.3f replay=%zu\n",
                 t + 1, steps, actual, K, policy_loss, value_loss, reward_loss,
                 latent_loss, w_min, w_sum / (float)actual, w_max, n);
        } else {
          printf("[train unroll] step=%d/%d batch=%d K=%d pol=%.6f val=%.6f "
                 "rew=%.6f lat=%.6f replay=%zu\n",
                 t + 1, steps, actual, K, policy_loss, value_loss, reward_loss,
                 latent_loss, n);
        }
      }
    }

    free(obs_seq);
    free(pi_seq);
    free(z_seq);
    free(vprefix_seq);
    free(a_seq);
    free(r_seq);
    free(done_seq);
    free(idx_out);
    free(prob_out);
    free(w_batch);
    return;
  }

  float *obs_batch = (float *)malloc(sizeof(float) * (size_t)B * (size_t)O);
  float *pi_batch = (float *)malloc(sizeof(float) * (size_t)B * (size_t)A);
  float *z_batch = (float *)malloc(sizeof(float) * (size_t)B);
  size_t *idx_out = (size_t *)malloc(sizeof(size_t) * (size_t)B);
  float *prob_out = (float *)malloc(sizeof(float) * (size_t)B);
  float *w_batch = (float *)malloc(sizeof(float) * (size_t)B);

  if (!obs_batch || !pi_batch || !z_batch || !idx_out || !prob_out ||
      !w_batch) {
    free(obs_batch);
    free(pi_batch);
    free(z_batch);
    free(idx_out);
    free(prob_out);
    free(w_batch);
    return;
  }

  // Optional: for logging only
  float *p_pred = (float *)malloc(sizeof(float) * (size_t)B * (size_t)A);
  float *v_pred = (float *)malloc(sizeof(float) * (size_t)B);

  for (int t = 0; t < steps; t++) {
    int actual = use_per ? rb_sample_per(rb, B, per_alpha, obs_batch, pi_batch,
                                         z_batch, idx_out, prob_out)
                         : rb_sample(rb, B, obs_batch, pi_batch, z_batch);
    if (actual <= 0)
      break;

    float w_min = 1.0f;
    float w_max = 1.0f;
    float w_sum = 0.0f;
    if (use_per) {
      const float N = (float)rb->size;
      float max_w = 1.0f;
      for (int i = 0; i < actual; i++) {
        float p = prob_out[i];
        float w = powf(N * p, -per_beta);
        w_batch[i] = w;
        if (w > max_w)
          max_w = w;
      }
      if (max_w > 0.0f) {
        float inv = 1.0f / max_w;
        for (int i = 0; i < actual; i++)
          w_batch[i] *= inv;
      }
      w_min = w_batch[0];
      w_max = w_batch[0];
      for (int i = 0; i < actual; i++) {
        float w = w_batch[i];
        if (w < w_min)
          w_min = w;
        if (w > w_max)
          w_max = w;
        w_sum += w;
      }
    } else {
      for (int i = 0; i < actual; i++)
        w_batch[i] = 1.0f;
      w_sum = (float)actual;
    }

    // Train policy/value via model hook or default batch trainer
    if (model->train_policy_value) {
      model->train_policy_value(model, obs_batch, pi_batch, z_batch, actual,
                                lr);
    } else {
      muzero_model_train_batch(model, obs_batch, pi_batch, z_batch, w_batch,
                               actual, lr);
    }

    // Lightweight logging (optional)
    if (p_pred && v_pred && (t == 0 || (t % 50) == 0)) {
      muzero_model_forward_batch(model, obs_batch, actual, p_pred, v_pred);

      double v_mse = 0.0;
      int cnt = 0;
      for (int i = 0; i < actual; i++) {
        float vp = v_pred[i];
        float zt = z_batch[i];
        if (is_finite_float(vp) && is_finite_float(zt)) {
          double d = (double)vp - (double)zt;
          v_mse += d * d;
          cnt++;
        }
      }
      if (cnt > 0)
        v_mse /= (double)cnt;

      if (use_per) {
        printf("[train pv] step=%d/%d batch=%d v_mse=%.6f "
               "w[min/mean/max]=%.3f/%.3f/%.3f replay=%zu\n",
               t + 1, steps, actual, (float)v_mse, w_min,
               w_sum / (float)actual, w_max, n);
      } else {
        printf("[train pv] step=%d/%d batch=%d v_mse=%.6f replay=%zu\n", t + 1,
               steps, actual, (float)v_mse, n);
      }
    }

    if (use_per && L > 0) {
      float *latent = (float *)malloc(sizeof(float) * (size_t)L);
      float *logits = (float *)malloc(sizeof(float) * (size_t)A);
      if (latent && logits) {
        for (int i = 0; i < actual; i++) {
          float v_pred = 0.0f;
          mu_model_repr(model, obs_batch + (size_t)i * (size_t)O, latent);
          mu_model_predict(model, latent, logits, &v_pred);
          float td = fabsf(v_pred - z_batch[i]);
          rb_set_priority(rb, idx_out[i], td + per_eps);
        }
      }
      free(latent);
      free(logits);
    }
  }

  free(p_pred);
  free(v_pred);

  free(obs_batch);
  free(pi_batch);
  free(z_batch);
  free(idx_out);
  free(prob_out);
  free(w_batch);
}
