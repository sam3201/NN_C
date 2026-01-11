#include "trainer.h"
#include "game_replay.h"
#include "muzero_model.h"
#include "replay_buffer.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static float priority_from_obs(MuModel *model, const float *obs,
                               const float *pi_tgt, float z_tgt, int A) {
  if (!model || !obs || !pi_tgt || A <= 0)
    return 0.0f;

  int L = model->cfg.latent_dim;
  float *latent = (float *)malloc(sizeof(float) * (size_t)L);
  float *logits = (float *)malloc(sizeof(float) * (size_t)A);
  if (!latent || !logits) {
    free(latent);
    free(logits);
    return 0.0f;
  }

  float v_pred = 0.0f;
  mu_model_repr(model, obs, latent);
  mu_model_predict(model, latent, logits, &v_pred);

  float pol_loss = cross_entropy_logits(pi_tgt, logits, A);
  float val_loss = mse(v_pred, z_tgt);

  free(latent);
  free(logits);
  return pol_loss + val_loss;
}

static int sample_game_sequence(GameReplay *gr, ReplayBuffer *rb, int batch,
                                int unroll_steps, float *obs_seq,
                                float *pi_seq, float *z_seq,
                                float *vprefix_seq, int *a_seq, float *r_seq,
                                int *done_seq, size_t *idx_out,
                                size_t *idx_seq, float per_alpha,
                                float *prob_out, size_t *seq_count_out) {
  if (!gr || !rb || !obs_seq || !pi_seq || !z_seq || !vprefix_seq || !a_seq ||
      !r_seq || !done_seq || !idx_out || !idx_seq)
    return 0;
  if (batch <= 0 || unroll_steps < 0)
    return 0;

  int O = gr->obs_dim;
  int A = gr->action_count;
  size_t steps = (size_t)unroll_steps + 1;

  size_t max_candidates = (size_t)gr->max_games * (size_t)gr->max_steps;
  size_t *start_offs = NULL;
  float *weights = NULL;
  float *cum = NULL;
  size_t candidates = 0;
  float total_w = 0.0f;

  start_offs = (size_t *)malloc(sizeof(size_t) * max_candidates);
  if (!start_offs)
    return 0;
  if (per_alpha > 0.0f) {
    weights = (float *)malloc(sizeof(float) * max_candidates);
    cum = (float *)malloc(sizeof(float) * max_candidates);
    if (!weights || !cum) {
      free(start_offs);
      free(weights);
      free(cum);
      return 0;
    }
  }

  for (int g = 0; g < gr->max_games; g++) {
    int T = gr->lengths[g];
    if (T <= (int)steps)
      continue;
    int max_start = T - (int)steps;
    for (int start = 0; start <= max_start; start++) {
      start_offs[candidates] =
          (size_t)g * (size_t)gr->max_steps + (size_t)start;
      if (per_alpha > 0.0f) {
        float pr_sum = 0.0f;
        for (size_t k = 0; k < steps; k++) {
          size_t off = ((size_t)g * (size_t)gr->max_steps +
                        (size_t)(start + (int)k));
          size_t rb_idx = gr->rb_idx_buf[off];
          float p = rb->prio_buf[rb_idx];
          if (!(p > 0.0f))
            p = 1e-6f;
          pr_sum += p;
        }
        float pr_mean = pr_sum / (float)steps;
        float w = powf(pr_mean, per_alpha);
        if (!(w > 0.0f))
          w = 1e-6f;
        weights[candidates] = w;
        total_w += w;
        cum[candidates] = total_w;
      }
      candidates++;
    }
  }

  if (seq_count_out)
    *seq_count_out = candidates;

  if (candidates == 0 || (per_alpha > 0.0f && !(total_w > 0.0f))) {
    free(start_offs);
    free(weights);
    free(cum);
    return 0;
  }

  int actual = 0;
  while (actual < batch) {
    size_t start_off = 0;
    float prob = 1.0f;
    if (per_alpha > 0.0f) {
      float u = ((float)rand() / (float)RAND_MAX) * total_w;
      size_t lo = 0;
      size_t hi = candidates - 1;
      while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (u <= cum[mid])
          hi = mid;
        else
          lo = mid + 1;
      }
      start_off = start_offs[lo];
      prob = weights[lo] / total_w;
    } else {
      size_t pick = (size_t)(rand() % (int)candidates);
      start_off = start_offs[pick];
      prob = 1.0f / (float)candidates;
    }
    int g = (int)(start_off / (size_t)gr->max_steps);
    int start = (int)(start_off % (size_t)gr->max_steps);

    for (size_t k = 0; k < steps; k++) {
      size_t off = ((size_t)g * (size_t)gr->max_steps + (size_t)(start + (int)k));
      memcpy(obs_seq + ((size_t)actual * steps + k) * (size_t)O,
             gr->obs_buf + off * (size_t)O, sizeof(float) * (size_t)O);
      memcpy(pi_seq + ((size_t)actual * steps + k) * (size_t)A,
             gr->pi_buf + off * (size_t)A, sizeof(float) * (size_t)A);

      size_t rb_idx = gr->rb_idx_buf[off];
      idx_seq[(size_t)actual * steps + k] = rb_idx;
      z_seq[(size_t)actual * steps + k] = rb->z_buf[rb_idx];
      vprefix_seq[(size_t)actual * steps + k] = rb->vprefix_buf[rb_idx];

      if (k < (size_t)unroll_steps) {
        a_seq[(size_t)actual * (size_t)unroll_steps + k] = gr->a_buf[off];
        r_seq[(size_t)actual * (size_t)unroll_steps + k] = gr->r_buf[off];
        done_seq[(size_t)actual * (size_t)unroll_steps + k] = gr->done_buf[off];
      }
    }
    idx_out[actual] =
        gr->rb_idx_buf[(size_t)g * (size_t)gr->max_steps + (size_t)start];
    if (prob_out)
      prob_out[actual] = prob;
    actual++;
  }

  free(start_offs);
  free(weights);
  free(cum);
  return actual;
}

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
  if (O <= 0)
    return;
  int L = model->cfg.latent_dim;

  float *obs_batch = (float *)malloc(sizeof(float) * (size_t)B * (size_t)O);
  float *next_obs_batch =
      (float *)malloc(sizeof(float) * (size_t)B * (size_t)O);
  int *a_batch = (int *)malloc(sizeof(int) * (size_t)B);
  float *r_batch = (float *)malloc(sizeof(float) * (size_t)B);
  int *done_batch = (int *)malloc(sizeof(int) * (size_t)B);
  size_t *idx_out = (size_t *)malloc(sizeof(size_t) * (size_t)B);
  float *prob_out = (float *)malloc(sizeof(float) * (size_t)B);
  float *w_batch = (float *)malloc(sizeof(float) * (size_t)B);

  if (!obs_batch || !next_obs_batch || !a_batch || !r_batch || !done_batch ||
      !idx_out || !prob_out || !w_batch) {
    free(obs_batch);
    free(next_obs_batch);
    free(a_batch);
    free(r_batch);
    free(done_batch);
    free(idx_out);
    free(prob_out);
    free(w_batch);
    return;
  }

  float sample_alpha = use_per ? per_alpha : 0.0f;
  for (int t = 0; t < steps; t++) {
    int actual = rb_sample_transition_per(
        rb, B, sample_alpha, obs_batch, a_batch, r_batch, next_obs_batch,
        done_batch, idx_out, prob_out);
    if (actual <= 0)
      break;

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
    } else {
      for (int i = 0; i < actual; i++)
        w_batch[i] = 1.0f;
    }

    if (cfg->reward_target_is_vprefix) {
      for (int i = 0; i < actual; i++) {
        r_batch[i] = rb->vprefix_buf[idx_out[i]];
      }
    }

    float lat_mse = 0.0f;
    float rew_mse = 0.0f;

    if (model->train_dynamics) {
      model->train_dynamics(model, obs_batch, a_batch, r_batch, next_obs_batch,
                            w_batch, cfg->train_reward_head, actual, lr,
                            &lat_mse, &rew_mse);
    } else {
      muzero_model_train_dynamics_batch(model, obs_batch, a_batch, r_batch,
                                        next_obs_batch, w_batch,
                                        cfg->train_reward_head, actual, lr,
                                        &lat_mse, &rew_mse);
    }

    if (t == 0 || (t % 50) == 0) {
      if (use_per) {
        float w_min = w_batch[0];
        float w_max = w_batch[0];
        float w_sum = 0.0f;
        for (int i = 0; i < actual; i++) {
          float w = w_batch[i];
          if (w < w_min)
            w_min = w;
          if (w > w_max)
            w_max = w;
          w_sum += w;
        }
        printf("[train dyn] step=%d/%d batch=%d latent_mse=%.6f reward_mse=%.6f "
               "w[min/mean/max]=%.3f/%.3f/%.3f replay=%zu\n",
               t + 1, steps, actual, lat_mse, rew_mse, w_min,
               w_sum / (float)actual, w_max, n);
      } else {
        printf("[train dyn] step=%d/%d batch=%d latent_mse=%.6f reward_mse=%.6f "
               "replay=%zu\n",
               t + 1, steps, actual, lat_mse, rew_mse, n);
      }
    }

    if (use_per && L > 0) {
      float *h = (float *)malloc(sizeof(float) * (size_t)L);
      float *h2 = (float *)malloc(sizeof(float) * (size_t)L);
      if (h && h2) {
        for (int i = 0; i < actual; i++) {
          const float *obs = obs_batch + (size_t)i * (size_t)O;
          mu_model_repr(model, obs, h);
          float r_pred = 0.0f;
          mu_model_dynamics(model, h, a_batch[i], h2, &r_pred);
          float td = fabsf(r_pred - r_batch[i]);
          rb_set_priority(rb, idx_out[i], td + per_eps);
        }
      }
      free(h);
      free(h2);
    }
  }

  free(obs_batch);
  free(next_obs_batch);
  free(a_batch);
  free(r_batch);
  free(done_batch);
  free(idx_out);
  free(prob_out);
  free(w_batch);
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

      if (model->train_unroll) {
        float *z_norm =
            (float *)malloc(sizeof(float) * (size_t)actual * steps_per);
        if (!z_norm)
          break;
        for (int i = 0; i < actual; i++) {
          for (size_t k = 0; k < steps_per; k++) {
            size_t idx = (size_t)i * steps_per + k;
            z_norm[idx] = mu_model_value_transform(model, z_seq[idx]);
          }
        }
        model->train_unroll(model, obs_seq, pi_seq, z_norm, vprefix_seq, a_seq,
                            r_seq, done_seq, w_batch, actual, K, n_boot,
                            discount, lr, &policy_loss, &value_loss,
                            &reward_loss, &latent_loss);
        free(z_norm);
      } else {
        muzero_model_train_unroll_batch(
            model, obs_seq, pi_seq, z_seq, vprefix_seq, a_seq, r_seq, done_seq,
            actual, K, n_boot, discount, lr, w_batch, &policy_loss,
            &value_loss, &reward_loss, &latent_loss);
      }

      if (use_per) {
        for (int i = 0; i < actual; i++) {
          const float *obs0 = obs_seq + (size_t)i * steps_per * (size_t)O;
          const float *pi0 = pi_seq + (size_t)i * steps_per * (size_t)A;
          float z_tgt = z_seq[(size_t)i * steps_per];
          float prio = priority_from_obs(model, obs0, pi0, z_tgt, A);
          rb_set_priority(rb, idx_out[i], prio + per_eps);
        }
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
      float *z_norm = (float *)malloc(sizeof(float) * (size_t)actual);
      if (!z_norm)
        break;
      for (int i = 0; i < actual; i++)
        z_norm[i] = mu_model_value_transform(model, z_batch[i]);
      model->train_policy_value(model, obs_batch, pi_batch, z_norm, w_batch,
                                actual, lr);
      free(z_norm);
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

    if (use_per) {
      for (int i = 0; i < actual; i++) {
        float prio =
            priority_from_obs(model, obs_batch + (size_t)i * (size_t)O,
                              pi_batch + (size_t)i * (size_t)A, z_batch[i], A);
        rb_set_priority(rb, idx_out[i], prio + per_eps);
      }
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

void trainer_train_from_replay_games(MuModel *model, ReplayBuffer *rb,
                                     GameReplay *gr,
                                     const TrainerConfig *cfg) {
  if (!model || !rb || !gr || !cfg)
    return;

  if (cfg->unroll_steps <= 0) {
    trainer_train_from_replay(model, rb, cfg);
    return;
  }

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
  if (O <= 0)
    return;
  int A = model->cfg.action_count;

  size_t steps_per = (size_t)K + 1;
  float *obs_seq =
      (float *)malloc(sizeof(float) * (size_t)B * steps_per * (size_t)O);
  float *pi_seq =
      (float *)malloc(sizeof(float) * (size_t)B * steps_per * (size_t)A);
  float *z_seq = (float *)malloc(sizeof(float) * (size_t)B * steps_per);
  float *vprefix_seq =
      (float *)malloc(sizeof(float) * (size_t)B * steps_per);
  int *a_seq = (int *)malloc(sizeof(int) * (size_t)B * (size_t)K);
  float *r_seq = (float *)malloc(sizeof(float) * (size_t)B * (size_t)K);
  int *done_seq = (int *)malloc(sizeof(int) * (size_t)B * (size_t)K);
  size_t *idx_out = (size_t *)malloc(sizeof(size_t) * (size_t)B);
  size_t *idx_seq =
      (size_t *)malloc(sizeof(size_t) * (size_t)B * steps_per);
  float *prob_out = (float *)malloc(sizeof(float) * (size_t)B);
  float *w_batch = (float *)malloc(sizeof(float) * (size_t)B);

  if (!obs_seq || !pi_seq || !z_seq || !vprefix_seq || !a_seq || !r_seq ||
      !done_seq || !idx_out || !idx_seq || !prob_out || !w_batch) {
    free(obs_seq);
    free(pi_seq);
    free(z_seq);
    free(vprefix_seq);
    free(a_seq);
    free(r_seq);
    free(done_seq);
    free(idx_out);
    free(idx_seq);
    free(prob_out);
    free(w_batch);
    return;
  }

  for (int t = 0; t < steps; t++) {
    size_t seq_count = 0;
    int actual = sample_game_sequence(gr, rb, B, K, obs_seq, pi_seq, z_seq,
                                      vprefix_seq, a_seq, r_seq, done_seq,
                                      idx_out, idx_seq,
                                      use_per ? per_alpha : 0.0f, prob_out,
                                      &seq_count);
    if (actual <= 0)
      break;

    if (use_per && seq_count > 0) {
      const float N = (float)seq_count;
      float max_w = 1.0f;
      for (int i = 0; i < actual; i++) {
        float p = prob_out[i];
        float w = (p > 0.0f) ? powf(N * p, -per_beta) : 1.0f;
        w_batch[i] = w;
        if (w > max_w)
          max_w = w;
      }
      if (max_w > 0.0f) {
        float inv = 1.0f / max_w;
        for (int i = 0; i < actual; i++)
          w_batch[i] *= inv;
      }
    } else {
      for (int i = 0; i < actual; i++)
        w_batch[i] = 1.0f;
    }

    float policy_loss = 0.0f;
    float value_loss = 0.0f;
    float reward_loss = 0.0f;
    float latent_loss = 0.0f;

    if (model->train_unroll) {
      float *z_norm =
          (float *)malloc(sizeof(float) * (size_t)actual * steps_per);
      if (!z_norm)
        break;
      for (int i = 0; i < actual; i++) {
        for (size_t k = 0; k < steps_per; k++) {
          size_t idx = (size_t)i * steps_per + k;
          z_norm[idx] = mu_model_value_transform(model, z_seq[idx]);
        }
      }
      model->train_unroll(model, obs_seq, pi_seq, z_norm, vprefix_seq, a_seq,
                          r_seq, done_seq, w_batch, actual, K, n_boot,
                          discount, lr, &policy_loss, &value_loss,
                          &reward_loss, &latent_loss);
      free(z_norm);
    } else {
      muzero_model_train_unroll_batch(
          model, obs_seq, pi_seq, z_seq, vprefix_seq, a_seq, r_seq, done_seq,
          actual, K, n_boot, discount, lr, w_batch, &policy_loss, &value_loss,
          &reward_loss, &latent_loss);
    }

    if (cfg->use_per) {
      float prio = policy_loss + value_loss + reward_loss + latent_loss;
      if (!is_finite_float(prio))
        prio = 0.0f;
      for (int i = 0; i < actual; i++) {
        for (size_t k = 0; k < steps_per; k++) {
          size_t rb_idx = idx_seq[(size_t)i * steps_per + k];
          rb_set_priority(rb, rb_idx, prio + per_eps);
        }
      }
    }

    if (t == 0 || (t % 50) == 0) {
      printf("[train unroll/games] step=%d/%d batch=%d K=%d pol=%.6f val=%.6f "
             "rew=%.6f lat=%.6f replay=%zu\n",
             t + 1, steps, actual, K, policy_loss, value_loss, reward_loss,
             latent_loss, n);
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
  free(idx_seq);
  free(prob_out);
  free(w_batch);
}
