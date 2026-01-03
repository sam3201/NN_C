#include "runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

MuRuntime *mu_runtime_create(MuModel *model, int capacity, float gamma) {
  MuRuntime *rt = calloc(1, sizeof(MuRuntime));
  rt->rb = rb_create(capacity, model->cfg.obs_dim, model->cfg.action_count);
  rt->gamma = gamma;
  rt->last_obs = malloc(sizeof(float) * model->cfg.obs_dim);
  rt->has_last = 0;
  return rt;
}

void mu_runtime_free(MuRuntime *rt) {
  if (!rt)
    return;
  rb_free(rt->rb);
  free(rt->last_obs);
  free(rt);
}

void mu_runtime_step(MuModel *model, MuRuntime *rt, const float *obs,
                     int action, float reward) {
  if (!rt->has_last) {
    memcpy(rt->last_obs, obs, sizeof(float) * model->cfg.obs_dim);
    rt->last_action = action;
    rt->has_last = 1;
    return;
  }

  /* Single-step target (bootstrap handled by MuZero value head) */
  float z = reward;

  float pi[model->cfg.action_count];
  for (int i = 0; i < model->cfg.action_count; i++)
    pi[i] = (i == action) ? 1.0f : 0.0f;

  rb_push(rt->rb, rt->last_obs, pi, z);

  memcpy(rt->last_obs, obs, sizeof(float) * model->cfg.obs_dim);
  rt->last_action = action;
}

void mu_runtime_end_episode(MuModel *model, MuRuntime *rt,
                            float terminal_reward) {
  if (!rt->has_last)
    return;

  float pi[model->cfg.action_count];
  memset(pi, 0, sizeof(pi));

  rb_push(rt->rb, rt->last_obs, pi, terminal_reward);
  rt->has_last = 0;
}

void mu_model_reset_episode(MuRuntime *rt) { rt->has_last = 0; }

void mu_model_train(MuModel *model, MuRuntime *rt) {
  if (rb_size(rt->rb) < 32)
    return;

  /* Placeholder SGD stub — you can upgrade later */
  printf("[MUZE] Training step (samples=%zu)\n", rb_size(rt->rb));
}

#include "runtime.h"

void mu_model_step(MuModel *m, const float *obs, int action, float reward) {
  mu_model_step(m, (MuRuntime *)m->runtime, obs, action, reward);
}

void mu_model_end_episode(MuModel *m, float terminal_reward) {
  mu_model_end_episode(m, (MuRuntime *)m->runtime, terminal_reward);
}

void mu_model_reset_episode(MuModel *m) {
  mu_model_reset_episode((MuRuntime *)m->runtime);
}

void mu_model_train(MuModel *m) { mu_model_train(m, (MuRuntime *)m->runtime); }
