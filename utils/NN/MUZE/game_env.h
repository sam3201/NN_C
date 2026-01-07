#pragma once
#include <stddef.h>

typedef struct {
  size_t obs_size;
  float *obs;
} ObsState;

typedef struct {
  // episode counters
  size_t step_i;
  size_t max_steps;

  // game state
  ObsState obs_state;

} GameEnv;

typedef void (*env_reset_fn)(ObsState *obs_state);
typedef int (*env_step_fn)(ObsState *obs_state, int action, float *reward,
                           int *done);

typedef struct {
  env_reset_fn reset;
  env_step_fn step;
} GameEnv;

void game_env_reset(ObsState *obs_state);
int game_env_step(ObsState *obs_state, int action, float *reward, int *done);
