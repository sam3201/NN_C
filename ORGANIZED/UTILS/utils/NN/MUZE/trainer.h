#ifndef MUZE_TRAINER_H
#define MUZE_TRAINER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "game_replay.h"
#include "muzero_model.h"
#include "replay_buffer.h"

typedef struct TrainerConfig {
  int batch_size;
  int train_steps;     // SGD steps per call
  int min_replay_size; // warmup threshold
  int unroll_steps;    // MuZero unroll steps
  int bootstrap_steps; // n-step bootstrap length
  float discount;      // bootstrap discount (gamma)
  int use_per;         // 0/1 prioritized replay
  float per_alpha;     // priority exponent
  float per_beta;      // importance-sampling exponent
  float per_beta_start; // schedule start (optional)
  float per_beta_end;   // schedule end (optional)
  int per_beta_anneal_steps; // schedule length in train steps
  float per_eps;       // small constant for priority
  int train_reward_head; // 0/1 train reward head in dynamics
  int reward_target_is_vprefix; // 0/1 use value prefix as reward target
  float lr;
} TrainerConfig;

void trainer_train_dynamics(MuModel *model, ReplayBuffer *rb,
                            const TrainerConfig *cfg);
void trainer_train_from_replay(MuModel *model, ReplayBuffer *rb,
                               const TrainerConfig *cfg);
void trainer_train_from_replay_games(MuModel *model, ReplayBuffer *rb,
                                     GameReplay *gr,
                                     const TrainerConfig *cfg);

#ifdef __cplusplus
}
#endif

#endif // MUZE_TRAINER_H
