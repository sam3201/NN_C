#ifndef SELFPLAY_H
#define SELFPLAY_H

#include "mcts.h"
#include "muzero_model.h"
#include "replay_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Environment callback signatures:
   - reset: should write initial observation into obs_out (length obs_dim)
   - step: given action, perform env step, write next obs to obs_out (length
   obs_dim), set *reward_out and *done_out (0/1). Return 0 on success. env_state
   is an opaque pointer to environment instance. */
typedef int (*env_step_fn)(void *env_state, int action, float *obs_out,
                           float *reward_out, int *done_out);

/* Self-play params */
typedef struct {
  int max_steps;      /* max steps per episode */
  float gamma;        /* discount for returns */
  float temperature;  /* sampling temperature during self-play (root) */
  int total_episodes; /* how many episodes to run */
} SelfPlayParams;

/* Run self-play episodes: each episode uses MCTS to choose actions (with
   MCTSParams) and pushes (obs, pi, z) samples into the replay buffer. env_state
   is user-provided and env callbacks operate on it.
*/
void selfplay_run(MuModel *model, void *env_state, env_reset_fn env_reset,
                  env_step_fn env_step, MCTSParams *mcts_params,
                  SelfPlayParams *sp_params, ReplayBuffer *rb);

#ifdef __cplusplus
}
#endif
#endif
