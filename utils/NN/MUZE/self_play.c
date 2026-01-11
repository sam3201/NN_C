#include "self_play.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void compute_discounted_returns(const float *rewards, int T, float gamma,
                                       float *z_out) {
  float acc = 0.0f;
  for (int t = T - 1; t >= 0; --t) {
    acc = rewards[t] + gamma * acc;
    z_out[t] = acc;
  }
}

static float lerp(float a, float b, float t) { return a + (b - a) * t; }

static float clampf(float x, float lo, float hi) {
  if (x < lo)
    return lo;
  if (x > hi)
    return hi;
  return x;
}

static int sample_from_probs_rng(const float *p, int n, MCTSRng *rng) {
  if (!p || n <= 0)
    return 0;

  float r = 0.0f;
  if (rng && rng->rand01)
    r = rng->rand01(rng->ctx);
  else
    r = (float)rand() / (float)RAND_MAX;

  float c = 0.0f;
  for (int i = 0; i < n; i++) {
    c += p[i];
    if (r <= c)
      return i;
  }
  return n - 1;
}

void selfplay_run(MuModel *model, void *env_state,
                  selfplay_env_reset_fn env_reset,
                  selfplay_env_step_fn env_step, MCTSParams *mcts_params,
                  SelfPlayParams *sp_params, ReplayBuffer *rb, GameReplay *gr,
                  MCTSRng *rng) {
  if (!model || !env_reset || !env_step || !mcts_params || !sp_params || !rb)
    return;

  int obs_dim = model->cfg.obs_dim;
  int A = model->cfg.action_count;
  int max_steps = sp_params->max_steps > 0 ? sp_params->max_steps : 200;

  float gamma = sp_params->gamma;
  if (!(gamma > 0.0f && gamma <= 1.0f))
    gamma = 0.997f;

  int log_every = sp_params->log_every > 0 ? sp_params->log_every : 10;

  float *obs_buf =
      (float *)malloc(sizeof(float) * (size_t)max_steps * (size_t)obs_dim);
  float *pi_buf =
      (float *)malloc(sizeof(float) * (size_t)max_steps * (size_t)A);
  float *reward_buf = (float *)malloc(sizeof(float) * (size_t)max_steps);
  int *act_buf = (int *)malloc(sizeof(int) * (size_t)max_steps);
  float *z_buf = (float *)malloc(sizeof(float) * (size_t)max_steps);

  float *obs0 = (float *)malloc(sizeof(float) * (size_t)obs_dim);

  size_t *idx_buf = (size_t *)malloc(sizeof(size_t) * (size_t)max_steps);

  if (!obs_buf || !pi_buf || !reward_buf || !act_buf || !obs0 || !idx_buf ||
      !z_buf) {
    free(obs_buf);
    free(pi_buf);
    free(reward_buf);
    free(act_buf);
    free(obs0);
    free(idx_buf);
    free(z_buf);
    return;
  }

  double avg_return = 0.0;
  double avg_root_v = 0.0;
  int avg_count = 0;

  for (int ep = 0; ep < sp_params->total_episodes; ep++) {
    env_reset(env_state, obs0);
    if (gr)
      gr_start_episode(gr);

    int step = 0;
    int done = 0;

    float *obs_cur = (float *)malloc(sizeof(float) * (size_t)obs_dim);
    if (!obs_cur)
      break;
    memcpy(obs_cur, obs0, sizeof(float) * (size_t)obs_dim);

    float ep_return = 0.0f;
    float ep_root_v_sum = 0.0f;

    while (!done && step < max_steps) {
      // ---- temperature schedule ----
      float t = 1.0f;
      if (sp_params->temp_decay_episodes > 0) {
        t = (float)ep / (float)sp_params->temp_decay_episodes;
      }
      t = clampf(t, 0.0f, 1.0f);

      float temp = lerp(sp_params->temp_start, sp_params->temp_end, t);
      if (!(temp > 0.0f))
        temp = 1e-6f;

      // ---- MCTS params for this step ----
      MCTSParams mp = *mcts_params;
      mp.temperature = temp;

      // Root noise during self-play (MuZero)
      if (sp_params->dirichlet_alpha > 0.0f &&
          sp_params->dirichlet_eps > 0.0f) {
        mp.dirichlet_alpha = sp_params->dirichlet_alpha;
        mp.dirichlet_eps = sp_params->dirichlet_eps;
      }

      // Run MCTS using rng
      MCTSResult mr = mcts_run(model, obs_cur, &mp, rng);
      ep_root_v_sum += mr.root_value;

      // Store obs and pi
      memcpy(obs_buf + (size_t)step * (size_t)obs_dim, obs_cur,
             sizeof(float) * (size_t)obs_dim);
      memcpy(pi_buf + (size_t)step * (size_t)A, mr.pi,
             sizeof(float) * (size_t)A);

      // Choose action by sampling MCTS policy (already temperature’d inside
      // mcts->pi)
      int chosen = sample_from_probs_rng(mr.pi, A, rng);
      act_buf[step] = chosen;

      // Step env
      float *next_obs = (float *)malloc(sizeof(float) * (size_t)obs_dim);
      if (!next_obs) {
        mcts_result_free(&mr);
        break;
      }

      float reward = 0.0f;
      int done_flag = 0;
      int ret = env_step(env_state, chosen, next_obs, &reward, &done_flag);
      if (ret != 0)
        done_flag = 1;

      // 2) also store (obs, pi, z) for policy/value training
      //    placeholder z for now; we will overwrite with discounted return
      //    later
      float z_placeholder = reward; // quick default
      idx_buf[step] = rb_push_full(rb, obs_cur, mr.pi, z_placeholder, chosen,
                                   reward, next_obs, done_flag);
      if (mr.root_value_dist && mr.root_value_bins > 0) {
        rb_set_value_dist(rb, idx_buf[step], mr.root_value_dist,
                          mr.root_value_bins);
      }
      if (gr) {
        gr_add_step(gr, obs_cur, mr.pi, chosen, reward, done_flag,
                    idx_buf[step]);
      }

      reward_buf[step] = reward;
      ep_return += reward;

      // Advance
      memcpy(obs_cur, next_obs, sizeof(float) * (size_t)obs_dim);
      free(next_obs);

      step++;
      if (done_flag)
        done = 1;

      mcts_result_free(&mr);
    }

    free(obs_cur);

    // ---- write proper discounted returns into replay (z targets) ----
    if (step > 0) {
      compute_discounted_returns(reward_buf, step, gamma, z_buf);
      float prefix = 0.0f;
      float gamma_pow = 1.0f;
      for (int i = 0; i < step; i++) {
        rb_set_z(rb, idx_buf[i], z_buf[i]);
        prefix += gamma_pow * reward_buf[i];
        rb_set_value_prefix(rb, idx_buf[i], prefix);
        gamma_pow *= gamma;
      }
    }
    if (gr)
      gr_end_episode(gr);

    // metrics
    float mean_root_v = (step > 0) ? (ep_root_v_sum / (float)step) : 0.0f;

    avg_return = 0.95 * avg_return + 0.05 * (double)ep_return;
    avg_root_v = 0.95 * avg_root_v + 0.05 * (double)mean_root_v;
    avg_count++;

    if ((ep % log_every) == 0) {
      printf("[selfplay] ep=%d steps=%d return=%.3f avg_return=%.3f rootV=%.3f "
             "avg_rootV=%.3f replay=%zu\n",
             ep, step, ep_return, (float)avg_return, mean_root_v,
             (float)avg_root_v, rb_size(rb));
    }
  }

  free(obs_buf);
  free(pi_buf);
  free(reward_buf);
  free(act_buf);
  free(obs0);
  free(z_buf);
  free(idx_buf);
}

void selfplay_run_threadsafe(MuModel *model, void *env_state,
                             selfplay_env_reset_fn env_reset,
                             selfplay_env_step_fn env_step,
                             MCTSParams *mcts_params,
                             SelfPlayParams *sp_params, ReplayBuffer *rb,
                             GameReplay *gr, MCTSRng *rng,
                             pthread_mutex_t *rb_mutex,
                             pthread_mutex_t *gr_mutex) {
  if (!model || !env_reset || !env_step || !mcts_params || !sp_params || !rb)
    return;

  int obs_dim = model->cfg.obs_dim;
  int A = model->cfg.action_count;
  int max_steps = sp_params->max_steps > 0 ? sp_params->max_steps : 200;

  float gamma = sp_params->gamma;
  if (!(gamma > 0.0f && gamma <= 1.0f))
    gamma = 0.997f;

  int log_every = sp_params->log_every > 0 ? sp_params->log_every : 10;

  float *obs_buf =
      (float *)malloc(sizeof(float) * (size_t)max_steps * (size_t)obs_dim);
  float *pi_buf =
      (float *)malloc(sizeof(float) * (size_t)max_steps * (size_t)A);
  float *reward_buf = (float *)malloc(sizeof(float) * (size_t)max_steps);
  int *act_buf = (int *)malloc(sizeof(int) * (size_t)max_steps);
  float *z_buf = (float *)malloc(sizeof(float) * (size_t)max_steps);

  float *obs0 = (float *)malloc(sizeof(float) * (size_t)obs_dim);

  size_t *idx_buf = (size_t *)malloc(sizeof(size_t) * (size_t)max_steps);

  if (!obs_buf || !pi_buf || !reward_buf || !act_buf || !obs0 || !idx_buf ||
      !z_buf) {
    free(obs_buf);
    free(pi_buf);
    free(reward_buf);
    free(act_buf);
    free(obs0);
    free(idx_buf);
    free(z_buf);
    return;
  }

  double avg_return = 0.0;
  double avg_root_v = 0.0;
  int avg_count = 0;

  for (int ep = 0; ep < sp_params->total_episodes; ep++) {
    env_reset(env_state, obs0);
    if (gr) {
      if (gr_mutex)
        pthread_mutex_lock(gr_mutex);
      gr_start_episode(gr);
      if (gr_mutex)
        pthread_mutex_unlock(gr_mutex);
    }

    int step = 0;
    int done = 0;

    float *obs_cur = (float *)malloc(sizeof(float) * (size_t)obs_dim);
    if (!obs_cur)
      break;
    memcpy(obs_cur, obs0, sizeof(float) * (size_t)obs_dim);

    float ep_return = 0.0f;
    float ep_root_v_sum = 0.0f;

    while (!done && step < max_steps) {
      float t = 1.0f;
      if (sp_params->temp_decay_episodes > 0) {
        t = (float)ep / (float)sp_params->temp_decay_episodes;
      }
      t = clampf(t, 0.0f, 1.0f);

      float temp = lerp(sp_params->temp_start, sp_params->temp_end, t);
      if (!(temp > 0.0f))
        temp = 1e-6f;

      MCTSParams mp = *mcts_params;
      mp.temperature = temp;

      if (sp_params->dirichlet_alpha > 0.0f &&
          sp_params->dirichlet_eps > 0.0f) {
        mp.dirichlet_alpha = sp_params->dirichlet_alpha;
        mp.dirichlet_eps = sp_params->dirichlet_eps;
      }

      MCTSResult mr = mcts_run(model, obs_cur, &mp, rng);
      ep_root_v_sum += mr.root_value;

      memcpy(obs_buf + (size_t)step * (size_t)obs_dim, obs_cur,
             sizeof(float) * (size_t)obs_dim);
      memcpy(pi_buf + (size_t)step * (size_t)A, mr.pi,
             sizeof(float) * (size_t)A);

      int chosen = sample_from_probs_rng(mr.pi, A, rng);
      act_buf[step] = chosen;

      float *next_obs = (float *)malloc(sizeof(float) * (size_t)obs_dim);
      if (!next_obs) {
        mcts_result_free(&mr);
        break;
      }

      float reward = 0.0f;
      int done_flag = 0;
      int ret = env_step(env_state, chosen, next_obs, &reward, &done_flag);
      if (ret != 0)
        done_flag = 1;

      float z_placeholder = reward;
      if (rb_mutex)
        pthread_mutex_lock(rb_mutex);
      idx_buf[step] = rb_push_full(rb, obs_cur, mr.pi, z_placeholder, chosen,
                                   reward, next_obs, done_flag);
      if (mr.root_value_dist && mr.root_value_bins > 0) {
        rb_set_value_dist(rb, idx_buf[step], mr.root_value_dist,
                          mr.root_value_bins);
      }
      if (rb_mutex)
        pthread_mutex_unlock(rb_mutex);

      if (gr) {
        if (gr_mutex)
          pthread_mutex_lock(gr_mutex);
        gr_add_step(gr, obs_cur, mr.pi, chosen, reward, done_flag,
                    idx_buf[step]);
        if (gr_mutex)
          pthread_mutex_unlock(gr_mutex);
      }

      reward_buf[step] = reward;
      ep_return += reward;

      memcpy(obs_cur, next_obs, sizeof(float) * (size_t)obs_dim);
      free(next_obs);

      step++;
      if (done_flag)
        done = 1;

      mcts_result_free(&mr);
    }

    free(obs_cur);

    if (step > 0) {
      compute_discounted_returns(reward_buf, step, gamma, z_buf);
      float prefix = 0.0f;
      float gamma_pow = 1.0f;
      for (int i = 0; i < step; i++) {
        if (rb_mutex)
          pthread_mutex_lock(rb_mutex);
        rb_set_z(rb, idx_buf[i], z_buf[i]);
        prefix += gamma_pow * reward_buf[i];
        rb_set_value_prefix(rb, idx_buf[i], prefix);
        if (rb_mutex)
          pthread_mutex_unlock(rb_mutex);
        gamma_pow *= gamma;
      }
    }
    if (gr) {
      if (gr_mutex)
        pthread_mutex_lock(gr_mutex);
      gr_end_episode(gr);
      if (gr_mutex)
        pthread_mutex_unlock(gr_mutex);
    }

    float mean_root_v = (step > 0) ? (ep_root_v_sum / (float)step) : 0.0f;

    avg_return = 0.95 * avg_return + 0.05 * (double)ep_return;
    avg_root_v = 0.95 * avg_root_v + 0.05 * (double)mean_root_v;
    avg_count++;

    if ((ep % log_every) == 0) {
      size_t replay_sz = 0;
      if (rb_mutex)
        pthread_mutex_lock(rb_mutex);
      replay_sz = rb_size(rb);
      if (rb_mutex)
        pthread_mutex_unlock(rb_mutex);

      printf("[selfplay] ep=%d steps=%d return=%.3f avg_return=%.3f rootV=%.3f "
             "avg_rootV=%.3f replay=%zu\n",
             ep, step, ep_return, (float)avg_return, mean_root_v,
             (float)avg_root_v, replay_sz);
    }
  }

  free(obs_buf);
  free(pi_buf);
  free(reward_buf);
  free(act_buf);
  free(obs0);
  free(z_buf);
  free(idx_buf);
}
