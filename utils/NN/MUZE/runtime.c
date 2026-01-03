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
