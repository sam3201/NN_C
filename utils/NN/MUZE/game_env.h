#pragma once
#include <stddef.h>

typedef struct {
  size_t obs_size;
  float *obs;
} GameState;

typedef struct {
  // episode counters
  size_t step_i;
  size_t max_steps;

  // game state
  GameState obs_state;

} GameEnv;

typedef void (*env_reset_fn)(GameState *obs_state);
typedef int (*env_step_fn)(GameState *obs_state, int action, float *reward,
                           int *done);

int game_env_step(GameState *obs_state, int action, float *reward, int *done);

GameEnv *game_env_init(GameEnv *game_env, env_reset_fn reset_fn,
                       env_step_fn step_fn);
void game_env_destroy(GameEnv *game_env);
void game_env_reset(GameState *obs_state);
int game_env_step(GameState *obs_state, int action, float *reward, int *done);
