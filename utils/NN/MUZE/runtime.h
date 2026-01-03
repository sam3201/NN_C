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

#endif
