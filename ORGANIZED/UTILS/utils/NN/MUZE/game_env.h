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

typedef void (*gameenv_reset_fn)(GameState *obs_state);
typedef int (*gameenv_step_fn)(GameState *obs_state, int action, float *reward,
                               int *done);

GameState *game_state_init(GameState *game_state, size_t obs_size);
void game_state_destroy(GameState *game_state);

GameEnv *game_env_init(GameEnv *game_env, gameenv_reset_fn reset_fn,
                       gameenv_step_fn step_fn);
void game_env_destroy(GameEnv *game_env);
void game_env_reset(GameState *obs_state);
int game_env_step(GameState *obs_state, int action, float *reward, int *done);
