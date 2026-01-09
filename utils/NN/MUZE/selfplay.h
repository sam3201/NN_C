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
typedef void (*selfplay_env_reset_fn)(void *env_state, float *obs_out);
typedef int (*selfplay_env_step_fn)(void *state, int action, float *obs,
                                    float *reward, int *done);

/* Self-play params */
typedef struct {
  int max_steps;
  float gamma;

  // Temperature scheduling:
  // - for early training, keep > 0 (explore)
  // - later anneal toward low values (exploit)
  float temp_start;        // e.g. 1.0
  float temp_end;          // e.g. 0.25
  int temp_decay_episodes; // e.g. 200

  // Root exploration noise (MuZero-style)
  float dirichlet_alpha; // e.g. 0.3 (for small action spaces)
  float dirichlet_eps;   // e.g. 0.25

  int total_episodes;

  // logging
  int log_every; // e.g. 10
} SelfPlayParams;

/* Run self-play episodes: each episode uses MCTS to choose actions (with
   MCTSParams) and pushes (obs, pi, z) samples into the replay buffer. env_state
   is user-provided and env callbacks operate on it.
*/
void selfplay_run(MuModel *model, void *env_state,
                  selfplay_env_reset_fn env_reset,
                  selfplay_env_step_fn env_step, MCTSParams *mcts_params,
                  SelfPlayParams *sp_params, ReplayBuffer *rb);

#ifdef __cplusplus
}
#endif
#endif
