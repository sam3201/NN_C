#ifndef MUZE_RUNTIME_H
#define MUZE_RUNTIME_H

#include "muzero_model.h"
#include "replay_buffer.h"

typedef struct {
  ReplayBuffer *rb;

  float *last_obs;
  int last_action;
  int has_last;

  float gamma;
} MuRuntime;

MuRuntime *mu_runtime_create(MuModel *model, int capacity, float gamma);
void mu_runtime_free(MuRuntime *rt);

void mu_model_step(MuModel *model, MuRuntime *rt, const float *obs, int action,
                   float reward);

void mu_model_end_episode(MuModel *model, MuRuntime *rt, float terminal_reward);

void mu_model_reset_episode(MuRuntime *rt);

/* Train using replay buffer */
void mu_model_train(MuModel *model, MuRuntime *rt);

#endif
