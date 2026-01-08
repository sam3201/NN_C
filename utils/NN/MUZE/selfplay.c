#include "selfplay.h"
#include "toy_env.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Helper: compute discounted returns for an episode of length T.
   rewards[0..T-1], gamma -> returns z_t = sum_{k=0..T-1-t} gamma^k *
   rewards[t+k] output z_out must be length T
*/
static void compute_discounted_returns(const float *rewards, int T, float gamma,
                                       float *z_out) {
  for (int t = 0; t < T; t++) {
    float acc = 0.0f;
    float g = 1.0f;
    for (int k = t; k < T; k++) {
      acc += g * rewards[k];
      g *= gamma;
    }
    z_out[t] = acc;
  }
}

/* Runs self-play episodes and pushes training tuples into replay buffer */
void selfplay_run(MuModel *model, void *env_state,
                  selfplay_env_reset_fn env_reset,
                  selfplay_env_step_fn env_step, MCTSParams *mcts_params,
                  SelfPlayParams *sp_params, ReplayBuffer *rb) {
  if (!model || !env_reset || !env_step || !mcts_params || !sp_params || !rb)
    return;

  int obs_dim = model->cfg.obs_dim;
  int A = model->cfg.action_count;
  int max_steps = sp_params->max_steps > 0 ? sp_params->max_steps : 200;

  /* temporary buffers for one episode */
  float *obs_buf = malloc(sizeof(float) * max_steps * obs_dim);
  float *pi_buf = malloc(sizeof(float) * max_steps * A);
  float *reward_buf = malloc(sizeof(float) * max_steps);
  int *act_buf = malloc(sizeof(int) * max_steps);

  for (int ep = 0; ep < sp_params->total_episodes; ep++) {
    /* reset env */
    float *obs0 = malloc(sizeof(float) * obs_dim);
    env_reset(env_state, obs0);

    int step = 0;
    int done = 0;
    float obs_cur[obs_dim];
    memcpy(obs_cur, obs0, sizeof(float) * obs_dim);

    while (!done && step < max_steps) {
      /* run MCTS for current obs */
      MCTSParams mp = *mcts_params;
      mp.temperature = sp_params->temperature > 0.0f ? sp_params->temperature
                                                     : mcts_params->temperature;

      MCTSResult mr = mcts_run(model, obs_cur, &mp, NULL);

      /* sample action according to pi (with rng) */
      float r = (float)rand() / (float)RAND_MAX;
      float cum = 0.0f;
      int chosen = 0;
      for (int a = 0; a < A; a++) {
        cum += mr.pi[a];
        if (r <= cum) {
          chosen = a;
          break;
        }
      }

      /* store obs and pi */
      memcpy(obs_buf + step * obs_dim, obs_cur, sizeof(float) * obs_dim);
      memcpy(pi_buf + step * A, mr.pi, sizeof(float) * A);

      /* step env */
      float next_obs[obs_dim];
      float reward = 0.0f;
      int done_flag = 0;
      int ret = env_step(env_state, chosen, next_obs, &reward, &done_flag);
      if (ret != 0) {
        /* env error: stop episode */
        done_flag = 1;
      }
      reward_buf[step] = reward;
      act_buf[step] = chosen;

      /* advance */
      memcpy(obs_cur, next_obs, sizeof(float) * obs_dim);
      step++;

      mcts_result_free(&mr);

      if (done_flag)
        done = 1;
    }

    /* compute discounted returns z_t and push samples to replay buffer */
    float *z = malloc(sizeof(float) * step);
    compute_discounted_returns(reward_buf, step, sp_params->gamma, z);
    for (int t = 0; t < step; t++) {
      rb_push(rb, obs_buf + t * obs_dim, pi_buf + t * A, z[t]);
    }
    free(z);
    free(obs0);
  }

  free(obs_buf);
  free(pi_buf);
  free(reward_buf);
  free(act_buf);
}
