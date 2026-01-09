// runtime.h

#ifndef MUZE_RUNTIME_H
#define MUZE_RUNTIME_H

#include "muze_cortex.h"
#include "muzero_model.h"
#include "replay_buffer.h"
#include "trainer.h"
#include <stdint.h>

#define TRAIN_WINDOW 1024         // training cache size, NOT memory size
#define TRAIN_WARMUP TRAIN_WINDOW // warmup cache size
typedef struct {
  ReplayBuffer *rb;

  float *last_obs;
  float *last_pi;
  int last_action;
  int has_last;

  float gamma;

  /* infinite logical memory */
  size_t total_steps;

  TrainerConfig tc;
} MuRuntime;

/* Runtime lifecycle */
MuRuntime *mu_runtime_create(MuModel *model, float gamma);
void mu_runtime_free(MuRuntime *rt);

/* Runtime operations (internal) */
void mu_runtime_step(MuRuntime *rt, MuModel *model, const float *obs,
                     int action, float reward);

void mu_runtime_step_with_pi(MuRuntime *rt, MuModel *model, const float *obs,
                             const float *pi, int action, float reward);

void mu_runtime_end_episode(MuRuntime *rt, MuModel *model,
                            float terminal_reward);

void mu_runtime_reset_episode(MuRuntime *rt);
void mu_runtime_train(MuRuntime *rt, MuModel *model);

int muze_select_action(MuCortex *cortex, const float *obs, size_t obs_dim,
                       float *out_pi, size_t action_count, MCTSRng *rng);

#endif
