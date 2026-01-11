#ifndef MUZE_LOOP_H
#define MUZE_LOOP_H

#include "game_replay.h"
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
  int reanalyze_full_games;       // 0/1 use full-game reanalyze when available

  // eval
  int eval_interval;   // iterations between eval runs
  int eval_episodes;   // eval episodes per run
  int eval_max_steps;  // max steps per eval episode

  // checkpointing
  int checkpoint_interval;      // iterations between checkpoints
  const char *checkpoint_prefix; // e.g. "checkpoints/muzero"
  int checkpoint_save_replay;   // 0/1 save replay buffer
  int checkpoint_save_games;    // 0/1 save game replay
  int checkpoint_keep_last;     // keep last N checkpoints (0 = keep all)
} MuLoopConfig;

/*
  Full MuZero-style loop:
    for iter:
      selfplay -> fill replay
      optionally reanalyze -> refresh (pi,z) targets for some samples
      train -> policy/value + dynamics/reward
*/
void muze_run_loop(MuModel *model, void *env_state,
                   selfplay_env_reset_fn env_reset,
                   selfplay_env_step_fn env_step, ReplayBuffer *rb,
                   GameReplay *gr,
                   const MCTSParams *base_mcts_params,
                   const SelfPlayParams *base_sp_params,
                   const MuLoopConfig *loop_cfg, MCTSRng *rng);

#ifdef __cplusplus
}
#endif

#endif
