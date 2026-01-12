#ifndef MUZE_LOOP_H
#define MUZE_LOOP_H

#include "game_replay.h"
#include "mcts.h"
#include "muzero_model.h"
#include "replay_buffer.h"
#include "self_play.h"
#include "trainer.h"
#include <pthread.h>

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

  // multi-actor selfplay
  int selfplay_actor_count; // number of actors (0/1 = single)
  int selfplay_use_threads; // 0/1 use pthreads when supported

  // reanalyze scheduling
  int reanalyze_interval; // iterations between reanalyze runs (0 = every iter)
  float reanalyze_fraction; // fraction of replay to reanalyze (0 = use samples)
  int reanalyze_min_replay; // minimum replay size for reanalyze

  // replay sharding
  int replay_shard_interval;  // iterations between shard saves
  int replay_shard_keep_last; // keep last N shards (0 = keep all)
  size_t replay_shard_max_entries; // compact to last N entries (0 = full)
  const char *replay_shard_prefix; // e.g. "replay/shard"
  int replay_shard_save_games; // 0/1 save game replay with shard

  // best model selection
  int eval_best_model;             // 0/1 save best model by eval score
  const char *best_checkpoint_prefix; // e.g. "checkpoints/muzero_best"
  int best_save_replay;            // 0/1 save replay for best
  int best_save_games;             // 0/1 save game replay for best
  int selfplay_disable;            // 0/1 disable selfplay even if eps set
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

void muze_run_loop_multi(MuModel *model, void *env_state,
                         selfplay_env_reset_fn env_reset,
                         selfplay_env_step_fn env_step,
                         selfplay_env_clone_fn env_clone,
                         selfplay_env_destroy_fn env_destroy,
                         ReplayBuffer *rb, GameReplay *gr,
                         const MCTSParams *base_mcts_params,
                         const SelfPlayParams *base_sp_params,
                         const MuLoopConfig *loop_cfg, MCTSRng *rng);

void muze_run_loop_locked(MuModel *model, void *env_state,
                          selfplay_env_reset_fn env_reset,
                          selfplay_env_step_fn env_step, ReplayBuffer *rb,
                          GameReplay *gr,
                          const MCTSParams *base_mcts_params,
                          const SelfPlayParams *base_sp_params,
                          const MuLoopConfig *loop_cfg, MCTSRng *rng,
                          pthread_mutex_t *model_mutex,
                          pthread_mutex_t *rb_mutex,
                          pthread_mutex_t *gr_mutex);

void muze_run_loop_multi_locked(MuModel *model, void *env_state,
                                selfplay_env_reset_fn env_reset,
                                selfplay_env_step_fn env_step,
                                selfplay_env_clone_fn env_clone,
                                selfplay_env_destroy_fn env_destroy,
                                ReplayBuffer *rb, GameReplay *gr,
                                const MCTSParams *base_mcts_params,
                                const SelfPlayParams *base_sp_params,
                                const MuLoopConfig *loop_cfg, MCTSRng *rng,
                                pthread_mutex_t *model_mutex,
                                pthread_mutex_t *rb_mutex,
                                pthread_mutex_t *gr_mutex);

#ifdef __cplusplus
}
#endif

#endif
