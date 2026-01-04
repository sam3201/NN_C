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

/* Runtime lifecycle */
MuRuntime *mu_runtime_create(MuModel *model, int capacity, float gamma);
void mu_runtime_free(MuRuntime *rt);

/* Internal runtime ops */
void mu_runtime_step(MuRuntime *rt, MuModel *model, const float *obs,
                     int action, float reward);

void mu_runtime_end_episode(MuRuntime *rt, MuModel *model,
                            float terminal_reward);
void mu_runtime_reset_episode(MuRuntime *rt);
void mu_runtime_train(MuRuntime *rt, MuModel *model);

#endif
