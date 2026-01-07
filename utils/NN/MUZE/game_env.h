#pragma once
#include <stddef.h>

typedef struct {
  size_t obs_size;
  float *obs;
} ObsState;

typedef struct {
  // episode counters
  int step_i;
  int max_steps;

  // copy of agent-like state (keep only what training needs)
  float x, y;
  float health, stamina;

  // base info (for reward / navigation)
  float base_x, base_y;
  float base_r;
  float base_integrity;

  float nearest_res_d[4];
  float nearest_res_dx[4];
  float nearest_res_dy[4];
  float nearest_res_alive[4];

  float nearest_mob_d, nearest_mob_dx, nearest_mob_dy, nearest_mob_type;
} GameEnvState;

typedef void (*env_reset_fn)(void *state, float *obs);
typedef int (*env_step_fn)(void *state, int action, float *obs, float *reward,
                           int *done);

void game_env_reset(void *state_ptr, float *obs);
int game_env_step(void *state_ptr, int action, float *obs, float *reward,
                  int *done);
