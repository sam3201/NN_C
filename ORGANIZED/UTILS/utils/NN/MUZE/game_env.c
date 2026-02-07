 #include "game_env.h"
 #include <stdlib.h>

 static gameenv_reset_fn g_reset_fn = NULL;
 static gameenv_step_fn g_step_fn = NULL;

 GameState *game_state_init(GameState *game_state, size_t obs_size) {
   if (!game_state)
     return NULL;
   game_state->obs_size = obs_size;
   game_state->obs = (float *)calloc(obs_size, sizeof(float));
   if (!game_state->obs)
     return NULL;
   return game_state;
 }

 void game_state_destroy(GameState *game_state) {
   if (!game_state)
     return;
   free(game_state->obs);
   game_state->obs = NULL;
   game_state->obs_size = 0;
 }

 GameEnv *game_env_init(GameEnv *game_env, gameenv_reset_fn reset_fn,
                        gameenv_step_fn step_fn) {
   if (!game_env)
     return NULL;
   game_env->step_i = 0;
   game_env->max_steps = 0;
   game_env->obs_state.obs_size = 0;
   game_env->obs_state.obs = NULL;
   g_reset_fn = reset_fn;
   g_step_fn = step_fn;
   return game_env;
 }

 void game_env_destroy(GameEnv *game_env) {
   if (!game_env)
     return;
   game_state_destroy(&game_env->obs_state);
 }

 void game_env_reset(GameState *obs_state) {
   if (!g_reset_fn)
     return;
   g_reset_fn(obs_state);
 }

 int game_env_step(GameState *obs_state, int action, float *reward, int *done) {
   if (!g_step_fn)
     return -1;
   return g_step_fn(obs_state, action, reward, done);
 }
