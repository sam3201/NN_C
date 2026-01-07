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

  float nearest_mob_d, nearest_mob_dx, nearest_mob_dy, nearest_mob_type;
} GameEnvState;

typedef void (*env_reset_fn)(void *state, float *obs);
typedef int (*env_step_fn)(void *state, int action, float *obs, float *reward,
                           int *done);

void game_env_reset(void *state_ptr, float *obs);
int game_env_step(void *state_ptr, int action, float *obs, float *reward,
                  int *done);
