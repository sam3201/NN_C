#include "muze_loop.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -------- helpers -------- */

static float clampf(float x, float lo, float hi) {
  if (x < lo)
    return lo;
  if (x > hi)
    return hi;
  return x;
}

static void normalize_probs(float *p, int n) {
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    if (!isfinite(p[i]) || p[i] < 0.0f)
      p[i] = 0.0f;
    sum += p[i];
  }
  if (sum <= 1e-12f) {
    float u = (n > 0) ? (1.0f / (float)n) : 0.0f;
    for (int i = 0; i < n; i++)
      p[i] = u;
    return;
  }
  float inv = 1.0f / sum;
  for (int i = 0; i < n; i++)
    p[i] *= inv;
}

static float cross_entropy_logits_local(const float *pi_target,
                                        const float *logits, int n) {
  float maxv = logits[0];
  for (int i = 1; i < n; i++)
    if (logits[i] > maxv)
      maxv = logits[i];

  float sum = 0.0f;
  for (int i = 0; i < n; i++)
    sum += expf(logits[i] - maxv);
  float logZ = logf(sum) + maxv;

  float loss = 0.0f;
  for (int i = 0; i < n; i++)
    loss -= pi_target[i] * (logits[i] - logZ);
  return loss;
}

static float priority_from_obs_local(MuModel *model, const float *obs,
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
  float pol_loss = cross_entropy_logits_local(pi_tgt, logits, A);
  float val_loss = (v_pred - z_tgt) * (v_pred - z_tgt);
  free(latent);
  free(logits);
  return pol_loss + val_loss;
}

static size_t rb_logical_to_physical_local(const ReplayBuffer *rb,
                                           size_t logical) {
  if (!rb)
    return 0;
  if (rb->size < rb->capacity)
    return logical;
  return (rb->write_idx + logical) % rb->capacity;
}

/*
  Reanalyze a single observation:
    - run MCTS with latest model
    - set pi_target = visit-count policy
    - set z_target = discounted return estimate:
        Here we can only do a lightweight version because ReplayBuffer stores
        per-step z already (from selfplay). So we refresh pi; optionally we can
        keep z as-is.
    - If you want true MuZero bootstrap targets, see notes below.
*/
static void reanalyze_one(MuModel *model, const float *obs, float *pi_out,
                          float *out_root_v, float *out_root_dist,
                          int dist_bins, const MCTSParams *mp,
                          MCTSRng *rng) {

  MCTSResult mr = mcts_run(model, obs, mp, rng);
  memcpy(pi_out, mr.pi, sizeof(float) * (size_t)mr.action_count);
  normalize_probs(pi_out, mr.action_count);

  if (out_root_v)
    *out_root_v = mr.root_value;
  if (out_root_dist && dist_bins > 0 && mr.root_value_dist &&
      mr.root_value_bins == dist_bins) {
    memcpy(out_root_dist, mr.root_value_dist,
           sizeof(float) * (size_t)dist_bins);
  }

  mcts_result_free(&mr);
}

/*
  Because your ReplayBuffer is a ring buffer of *independent* samples,
  we reanalyze by:
    - sampling a random batch of observations from rb->obs_buf
    - running MCTS on each
    - overwriting rb->pi_buf for those slots (and optionally rb->z_buf)
*/
static void reanalyze_replay(MuModel *model, ReplayBuffer *rb,
                             const MCTSParams *mp, int samples, int unroll,
                             int bootstrap_steps, float gamma, MCTSRng *rng) {

  if (!model || !rb || rb->size == 0 || samples <= 0)
    return;

  const int O = rb->obs_dim;
  const int A = rb->action_count;
  if (unroll < 0)
    unroll = 0;

  float *pi_tmp = (float *)malloc(sizeof(float) * (size_t)A);
  float *root_v = (float *)malloc(sizeof(float) * (size_t)(unroll + 1));
  if (!pi_tmp)
    return;

  double avg_v = 0.0;
  int cnt = 0;

  size_t need = (size_t)unroll + 1;
  if (rb->size < need) {
    free(pi_tmp);
    free(root_v);
    return;
  }

  size_t max_start = rb->size - need;
  for (int s = 0; s < samples; s++) {
    size_t start_logical = 0;
    if (rng && rng->rand01) {
      start_logical = (size_t)(rng->rand01(rng->ctx) * (float)(max_start + 1));
    } else {
      start_logical = (size_t)((double)rand() /
                               ((double)RAND_MAX + 1.0) * (double)(max_start + 1));
    }

    for (int k = 0; k <= unroll; k++) {
      size_t logical = start_logical + (size_t)k;
      size_t idx = rb_logical_to_physical_local(rb, logical);
      const float *obs = rb->obs_buf + idx * (size_t)O;
      reanalyze_one(model, obs, pi_tmp, &root_v[k],
                    (rb->value_dist_buf && rb->support_size > 1)
                        ? rb->value_dist_buf + idx * (size_t)rb->support_size
                        : NULL,
                    rb->support_size, mp, rng);
      memcpy(rb->pi_buf + idx * (size_t)A, pi_tmp,
             sizeof(float) * (size_t)A);
    }

    // recompute value prefix + bootstrap targets
    for (int k = 0; k <= unroll; k++) {
      size_t logical = start_logical + (size_t)k;
      size_t idx = rb_logical_to_physical_local(rb, logical);

      float prefix = 0.0f;
      float gamma_pow = 1.0f;
      for (int i = 0; i < k; i++) {
        size_t li = start_logical + (size_t)i;
        size_t ii = rb_logical_to_physical_local(rb, li);
        prefix += gamma_pow * rb->r_buf[ii];
        gamma_pow *= gamma;
        if (rb->done_buf[ii])
          break;
      }
      rb->vprefix_buf[idx] = prefix;

      float G = 0.0f;
      gamma_pow = 1.0f;
      int done_flag = 0;
      int max_i = bootstrap_steps;
      for (int i = 0; i < max_i; i++) {
        int t = k + i;
        if (t > unroll)
          break;
        size_t li = start_logical + (size_t)t;
        size_t ii = rb_logical_to_physical_local(rb, li);
        G += gamma_pow * rb->r_buf[ii];
        gamma_pow *= gamma;
        if (rb->done_buf[ii]) {
          done_flag = 1;
          break;
        }
      }
      float bootstrap = 0.0f;
      if (!done_flag) {
        int t_boot = k + max_i;
        if (t_boot > unroll)
          t_boot = unroll;
        bootstrap = root_v[t_boot];
      }
      rb->z_buf[idx] = G + gamma_pow * bootstrap;

      const float *obs = rb->obs_buf + idx * (size_t)O;
      float prio =
          priority_from_obs_local(model, obs, rb->pi_buf + idx * (size_t)A,
                                  rb->z_buf[idx], A);
      rb_set_priority(rb, idx, prio + 1e-3f);

      avg_v += root_v[k];
      cnt++;
    }
  }

  if (cnt > 0) {
    avg_v /= (double)cnt;
    printf("[reanalyze] samples=%d avg_root_v=%.4f\n", cnt, (float)avg_v);
  }

  free(pi_tmp);
  free(root_v);
}

static void reanalyze_games(MuModel *model, ReplayBuffer *rb, GameReplay *gr,
                            const MCTSParams *mp, int bootstrap_steps,
                            float gamma, MCTSRng *rng) {
  if (!model || !rb || !gr || gr->game_count == 0)
    return;
  if (!(gamma > 0.0f))
    gamma = 0.997f;
  if (bootstrap_steps <= 0)
    bootstrap_steps = 1;

  const int O = gr->obs_dim;
  const int A = gr->action_count;

  float *pi_tmp = (float *)malloc(sizeof(float) * (size_t)A);
  float *root_v = (float *)malloc(sizeof(float) * (size_t)gr->max_steps);
  if (!pi_tmp || !root_v) {
    free(pi_tmp);
    free(root_v);
    return;
  }

  double avg_v = 0.0;
  int count = 0;

  for (int g = 0; g < gr->max_games; g++) {
    int T = gr->lengths[g];
    if (T <= 0)
      continue;

    for (int t = 0; t < T; t++) {
      size_t off = ((size_t)g * (size_t)gr->max_steps + (size_t)t);
      const float *obs = gr->obs_buf + off * (size_t)O;
      size_t rb_idx = gr->rb_idx_buf[off];
      float *dist_out = NULL;
      if (rb_idx < rb->capacity && rb->value_dist_buf && rb->support_size > 1) {
        dist_out =
            rb->value_dist_buf + rb_idx * (size_t)rb->support_size;
      }

      reanalyze_one(model, obs, pi_tmp, &root_v[t],
                    dist_out, rb->support_size, mp, rng);
      memcpy(gr->pi_buf + off * (size_t)A, pi_tmp,
             sizeof(float) * (size_t)A);

      if (rb_idx < rb->capacity) {
        memcpy(rb->pi_buf + rb_idx * (size_t)A, pi_tmp,
               sizeof(float) * (size_t)A);
      }
      avg_v += root_v[t];
      count++;
    }

    float prefix = 0.0f;
    float gamma_pow_prefix = 1.0f;
    for (int t = 0; t < T; t++) {
      size_t off = ((size_t)g * (size_t)gr->max_steps + (size_t)t);
      prefix += gamma_pow_prefix * gr->r_buf[off];
      gamma_pow_prefix *= gamma;

      size_t rb_idx = gr->rb_idx_buf[off];
      rb_set_value_prefix(rb, rb_idx, prefix);
    }

    for (int t = 0; t < T; t++) {
      float G = 0.0f;
      float gamma_pow = 1.0f;
      int done_flag = 0;

      for (int i = 0; i < bootstrap_steps; i++) {
        int ti = t + i;
        if (ti >= T)
          break;
        size_t off = ((size_t)g * (size_t)gr->max_steps + (size_t)ti);
        G += gamma_pow * gr->r_buf[off];
        gamma_pow *= gamma;
        if (gr->done_buf[off]) {
          done_flag = 1;
          break;
        }
      }

      float bootstrap = 0.0f;
      int t_boot = t + bootstrap_steps;
      if (!done_flag && t_boot < T)
        bootstrap = root_v[t_boot];

      size_t off = ((size_t)g * (size_t)gr->max_steps + (size_t)t);
      size_t rb_idx = gr->rb_idx_buf[off];
      rb_set_z(rb, rb_idx, G + gamma_pow * bootstrap);
      float prio =
          priority_from_obs_local(model, gr->obs_buf + off * (size_t)O,
                                  rb->pi_buf + rb_idx * (size_t)A,
                                  rb->z_buf[rb_idx], A);
      rb_set_priority(rb, rb_idx, prio + 1e-3f);
    }
  }

  if (count > 0) {
    avg_v /= (double)count;
    printf("[reanalyze] games=%d avg_root_v=%.4f\n", gr->game_count,
           (float)avg_v);
  }

  free(pi_tmp);
  free(root_v);
}

static float eval_run(MuModel *model, void *env_state,
                      selfplay_env_reset_fn env_reset,
                      selfplay_env_step_fn env_step,
                      const MCTSParams *base_mp, int episodes,
                      int max_steps, float gamma, MCTSRng *rng) {
  if (!model || !env_reset || !env_step || !base_mp || episodes <= 0)
    return 0.0f;

  if (!(gamma > 0.0f))
    gamma = 0.997f;
  if (max_steps <= 0)
    max_steps = 200;

  int A = model->cfg.action_count;
  double sum_return = 0.0;
  double sum_steps = 0.0;
  float min_return = 0.0f;
  float max_return = 0.0f;
  int min_steps = 0;
  int max_steps_seen = 0;

  float *obs = (float *)malloc(sizeof(float) * (size_t)model->cfg.obs_dim);
  if (!obs)
    return 0.0f;

  for (int ep = 0; ep < episodes; ep++) {
    env_reset(env_state, obs);

    float ep_return = 0.0f;
    int steps = 0;
    int done = 0;

    while (!done && steps < max_steps) {
      MCTSParams mp = *base_mp;
      mp.dirichlet_alpha = 0.0f;
      mp.dirichlet_eps = 0.0f;
      mp.temperature = 0.0f;

      MCTSResult mr = mcts_run(model, obs, &mp, rng);

      int best = 0;
      float bestp = mr.pi[0];
      for (int a = 1; a < A; a++) {
        if (mr.pi[a] > bestp) {
          bestp = mr.pi[a];
          best = a;
        }
      }

      float *next_obs =
          (float *)malloc(sizeof(float) * (size_t)model->cfg.obs_dim);
      if (!next_obs) {
        mcts_result_free(&mr);
        break;
      }

      float reward = 0.0f;
      int done_flag = 0;
      int ret = env_step(env_state, best, next_obs, &reward, &done_flag);
      if (ret != 0)
        done_flag = 1;

      ep_return += reward;
      memcpy(obs, next_obs, sizeof(float) * (size_t)model->cfg.obs_dim);
      free(next_obs);

      steps++;
      done = done_flag;
      mcts_result_free(&mr);
    }

    if (ep == 0) {
      min_return = max_return = ep_return;
      min_steps = max_steps_seen = steps;
    } else {
      if (ep_return < min_return)
        min_return = ep_return;
      if (ep_return > max_return)
        max_return = ep_return;
      if (steps < min_steps)
        min_steps = steps;
      if (steps > max_steps_seen)
        max_steps_seen = steps;
    }
    sum_return += (double)ep_return;
    sum_steps += (double)steps;
  }

  free(obs);
  float mean_return = (episodes > 0) ? (float)(sum_return / episodes) : 0.0f;
  float mean_steps = (episodes > 0) ? (float)(sum_steps / episodes) : 0.0f;
  printf("[eval] episodes=%d mean_return=%.3f min/max=%.3f/%.3f "
         "mean_steps=%.2f min/max=%d/%d\n",
         episodes, mean_return, min_return, max_return, mean_steps, min_steps,
         max_steps_seen);
  return mean_return;
}

/* -------- public loop -------- */

void muze_run_loop(MuModel *model, void *env_state,
                   selfplay_env_reset_fn env_reset,
                   selfplay_env_step_fn env_step, ReplayBuffer *rb,
                   GameReplay *gr,
                   const MCTSParams *base_mcts_params,
                   const SelfPlayParams *base_sp_params,
                   const MuLoopConfig *loop_cfg, MCTSRng *rng) {

  if (!model || !rb || !env_reset || !env_step || !base_mcts_params ||
      !base_sp_params || !loop_cfg)
    return;

  MCTSParams mp0 = *base_mcts_params;
  SelfPlayParams sp0 = *base_sp_params;

  int iters = loop_cfg->iterations > 0 ? loop_cfg->iterations : 1;
  int eps_per = loop_cfg->selfplay_episodes_per_iter > 0
                    ? loop_cfg->selfplay_episodes_per_iter
                    : 10;
  int train_calls =
      loop_cfg->train_calls_per_iter > 0 ? loop_cfg->train_calls_per_iter : 1;

  TrainerConfig tc = loop_cfg->train_cfg;

  // If user didn’t set these, keep sane defaults
  if (tc.batch_size <= 0)
    tc.batch_size = 32;
  if (tc.train_steps <= 0)
    tc.train_steps = 200;
  if (tc.min_replay_size <= 0)
    tc.min_replay_size = 1024;
  if (tc.unroll_steps <= 0)
    tc.unroll_steps = 5;
  if (tc.bootstrap_steps <= 0)
    tc.bootstrap_steps = tc.unroll_steps;
  if (tc.discount <= 0.0f)
    tc.discount = 0.997f;
  if (tc.per_alpha <= 0.0f)
    tc.per_alpha = 0.6f;
  if (tc.per_beta <= 0.0f)
    tc.per_beta = 0.4f;
  if (tc.per_beta_start <= 0.0f)
    tc.per_beta_start = tc.per_beta;
  if (tc.per_beta_end <= 0.0f)
    tc.per_beta_end = 1.0f;
  if (tc.per_eps <= 0.0f)
    tc.per_eps = 1e-3f;
  if (tc.train_reward_head < 0)
    tc.train_reward_head = 0;
  if (tc.reward_target_is_vprefix == 0 && tc.train_reward_head == 0)
    tc.reward_target_is_vprefix = 1;
  if (tc.lr <= 0.0f)
    tc.lr = 0.05f;

  int per_beta_step = 0;

  for (int it = 0; it < iters; it++) {
    // ---- self-play ----
    SelfPlayParams sp = sp0;
    sp.total_episodes = eps_per;

    printf("\n=== [loop] iter=%d/%d selfplay_episodes=%d ===\n", it + 1, iters,
           eps_per);
    selfplay_run(model, env_state, env_reset, env_step, &mp0, &sp, rb, gr, rng);

    // ---- reanalyze (optional) ----
    if (loop_cfg->use_reanalyze) {
      int samples = loop_cfg->reanalyze_samples_per_iter;
      if (samples <= 0)
        samples = 256;

      // Use a “training-time” MCTS setup: usually no root dirichlet during
      // reanalyze, and lower temperature. But you can keep it.
      MCTSParams mp = mp0;
      mp.dirichlet_alpha = 0.0f;
      mp.dirichlet_eps = 0.0f;
      mp.temperature =
          1.0f; // store full visit distribution; don’t harden too much

      float rgamma =
          (loop_cfg->reanalyze_gamma > 0.0f) ? loop_cfg->reanalyze_gamma
                                             : tc.discount;
      if (loop_cfg->reanalyze_full_games && gr && gr->game_count > 0) {
        reanalyze_games(model, rb, gr, &mp, tc.bootstrap_steps, rgamma, rng);
      } else {
        reanalyze_replay(model, rb, &mp, samples, tc.unroll_steps,
                         tc.bootstrap_steps, rgamma, rng);
      }
    }

    // ---- training ----
    printf("=== [loop] train_calls=%d ===\n", train_calls);
    for (int k = 0; k < train_calls; k++) {
      if (tc.per_beta_anneal_steps > 0) {
        float t = (float)per_beta_step / (float)tc.per_beta_anneal_steps;
        if (t > 1.0f)
          t = 1.0f;
        tc.per_beta =
            tc.per_beta_start + t * (tc.per_beta_end - tc.per_beta_start);
        per_beta_step++;
      }
      if (gr && tc.unroll_steps > 0)
        trainer_train_from_replay_games(model, rb, gr, &tc);
      else
        trainer_train_from_replay(model, rb, &tc);
      if (tc.unroll_steps <= 0)
        trainer_train_dynamics(model, rb, &tc);
    }

    printf("=== [loop] iter=%d done replay=%zu ===\n", it + 1, rb_size(rb));

    if (loop_cfg->eval_interval > 0 &&
        ((it + 1) % loop_cfg->eval_interval) == 0) {
      int eval_eps = loop_cfg->eval_episodes > 0 ? loop_cfg->eval_episodes : 10;
      int eval_steps =
          loop_cfg->eval_max_steps > 0 ? loop_cfg->eval_max_steps : sp0.max_steps;
      eval_run(model, env_state, env_reset, env_step, &mp0, eval_eps, eval_steps,
               sp0.gamma, rng);
    }

    if (loop_cfg->checkpoint_interval > 0 &&
        loop_cfg->checkpoint_prefix &&
        ((it + 1) % loop_cfg->checkpoint_interval) == 0) {
      char path[512];
      snprintf(path, sizeof(path), "%s_model_%04d.bin",
               loop_cfg->checkpoint_prefix, it + 1);
      mu_model_save(model, path);

      if (loop_cfg->checkpoint_save_replay) {
        snprintf(path, sizeof(path), "%s_replay_%04d.bin",
                 loop_cfg->checkpoint_prefix, it + 1);
        rb_save(rb, path);
      }
      if (loop_cfg->checkpoint_save_games && gr) {
        snprintf(path, sizeof(path), "%s_games_%04d.bin",
                 loop_cfg->checkpoint_prefix, it + 1);
        gr_save(gr, path);
      }

      if (loop_cfg->checkpoint_keep_last > 0) {
        int keep = loop_cfg->checkpoint_keep_last;
        int old_iter = (it + 1) - keep * loop_cfg->checkpoint_interval;
        if (old_iter > 0) {
          snprintf(path, sizeof(path), "%s_model_%04d.bin",
                   loop_cfg->checkpoint_prefix, old_iter);
          remove(path);
          if (loop_cfg->checkpoint_save_replay) {
            snprintf(path, sizeof(path), "%s_replay_%04d.bin",
                     loop_cfg->checkpoint_prefix, old_iter);
            remove(path);
          }
          if (loop_cfg->checkpoint_save_games && gr) {
            snprintf(path, sizeof(path), "%s_games_%04d.bin",
                     loop_cfg->checkpoint_prefix, old_iter);
            remove(path);
          }
        }
      }
    }
  }
}
