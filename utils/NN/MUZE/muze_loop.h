#ifndef MUZE_LOOP_H
#define MUZE_LOOP_H

#include "mcts.h"
#include "muzero_model.h"
#include "replay_buffer.h"
#include "self_play.h"
#include "trainer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  // outer loop
  int iterations;

  // selfplay
  int selfplay_episodes_per_iter;

  // training
  int train_calls_per_iter; // how many times to call trainer per iter
  TrainerConfig train_cfg;

  // reanalyze
  int use_reanalyze;              // 0/1
  int reanalyze_samples_per_iter; // number of random states to reanalyze
  float reanalyze_gamma;          // discount for z recompute
} MuLoopConfig;

/*
  Full MuZero-style loop:
    for iter:
      selfplay -> fill replay
      optionally reanalyze -> refresh (pi,z) targets for some samples
      train -> policy/value + dynamics/reward
*/
void muze_run_loop(MuModel *model, void *env_state,
                   self_play_env_reset_fn env_reset,
                   selfplay_env_step_fn env_step, ReplayBuffer *rb,
                   const MCTSParams *base_mcts_params,
                   const SelfPlayParams *base_sp_params,
                   const MuLoopConfig *loop_cfg, MCTSRng *rng);

#ifdef __cplusplus
}
#endif

#endif
