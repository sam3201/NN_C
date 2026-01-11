#ifndef MUZE_CONFIG_H
#define MUZE_CONFIG_H

#include "mcts.h"
#include "muze_loop.h"
#include "muzero_model.h"
#include "self_play.h"
#include "trainer.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int actor_count;
  int use_threads;
} MuzeActorConfig;

typedef struct {
  int shard_interval;
  int shard_keep_last;
  size_t shard_max_entries;
  const char *shard_prefix;
  int shard_save_games;
} MuzeReplayShardConfig;

typedef struct {
  int interval;
  float fraction;
  int min_replay_size;
} MuzeReanalyzeSchedule;

typedef struct {
  int eval_best_model;
  const char *best_checkpoint_prefix;
  int best_save_replay;
  int best_save_games;
} MuzeBestModelConfig;

typedef struct {
  uint32_t seed;
} MuzeSeedConfig;

typedef struct {
  MuConfig model;
  MuNNConfig nn;
  MCTSParams mcts;
  SelfPlayParams selfplay;
  TrainerConfig trainer;
  MuLoopConfig loop;
  MuzeActorConfig actors;
  MuzeReplayShardConfig replay_shards;
  MuzeReanalyzeSchedule reanalyze;
  MuzeBestModelConfig best;
  MuzeSeedConfig seed;
} MuzeConfig;

#ifdef __cplusplus
}
#endif

#endif
