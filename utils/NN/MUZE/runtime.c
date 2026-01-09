#include "runtime.h"
#include "trainer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

MuRuntime *mu_runtime_create(MuModel *model, float gamma) {
  MuRuntime *rt = calloc(1, sizeof(MuRuntime));

  rt->rb = rb_create(TRAIN_WINDOW, model->cfg.obs_dim, model->cfg.action_count);

  rt->last_obs = malloc(sizeof(float) * model->cfg.obs_dim);
  rt->has_last = 0;
  rt->gamma = gamma;
  rt->total_steps = 0;

  return rt;
}

void mu_runtime_free(MuRuntime *rt) {
  if (!rt)
    return;
  rb_free(rt->rb);
  free(rt->last_obs);
  free(rt);
}

void mu_runtime_step(MuRuntime *rt, MuModel *model, const float *obs,
                     int action, float reward) {
  rt->total_steps++;

  if (!rt->has_last) {
    memcpy(rt->last_obs, obs, sizeof(float) * model->cfg.obs_dim);
    rt->last_action = action;
    rt->has_last = 1;
    return;
  }

  /* One-step bootstrap target */
  float z = reward;

  float pi[model->cfg.action_count];
  for (int i = 0; i < model->cfg.action_count; i++)
    pi[i] = (i == rt->last_action) ? 1.0f : 0.0f;

  rb_push(rt->rb, rt->last_obs, pi, z);

  memcpy(rt->last_obs, obs, sizeof(float) * model->cfg.obs_dim);
  rt->last_action = action;
}

void mu_runtime_end_episode(MuRuntime *rt, MuModel *model,
                            float terminal_reward) {
  if (!rt->has_last)
    return;

  float pi[model->cfg.action_count];
  memset(pi, 0, sizeof(pi));

  rb_push(rt->rb, rt->last_obs, pi, terminal_reward);
  rt->has_last = 0;
}

void mu_runtime_reset_episode(MuRuntime *rt) { rt->has_last = 0; }

void mu_runtime_train(MuRuntime *rt, MuModel *model) {
  if (!rt || !model)
    return;

  // you can tune these defaults later
  TrainerConfig tc = {
      .batch_size = 32,
      .train_steps = 200, // smaller per call; call more often
      .min_replay_size = 128,
      .lr = 0.05f,
  };

  trainer_train_from_replay(model, rt->rb, &tc);
}

// Chooses action + fills out_pi[A].
// If cortex->use_mcts, uses MCTS (requires mcts_model).
// Else uses cortex->encode + cortex->policy.
// Applies policy_temperature + policy_epsilon.
int muze_select_action(MuCortex *cortex, const float *obs, size_t obs_dim,
                       float *out_pi, size_t action_count, MCTSRng *rng) {
  if (cortex->use_mcts && cortex->mcts_model) {
    MCTSResult mr = mcts_run(cortex->mcts_model, obs, cortex->mcts_params, rng)
                        

                    &cortex->mcts_params, rng) long double * *
               latent_seq = NULL;
    size_t seq_len = 0;
  }
