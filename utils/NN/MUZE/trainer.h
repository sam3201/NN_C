#ifndef MUZE_TRAINER_H
#define MUZE_TRAINER_H

#ifdef __cplusplus
extern "C" {
#endif

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
  float per_eps;       // small constant for priority
  float lr;
} TrainerConfig;

void trainer_train_dynamics(MuModel *model, ReplayBuffer *rb,
                            const TrainerConfig *cfg);
void trainer_train_from_replay(MuModel *model, ReplayBuffer *rb,
                               const TrainerConfig *cfg);

#ifdef __cplusplus
}
#endif

#endif // MUZE_TRAINER_H
