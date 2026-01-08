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
  float lr;
} TrainerConfig;

vvoid trainer_train_dynamics(MuModel *model, ReplayBuffer *rb,
                             const TrainerConfig *cfg);
oid trainer_train_from_replay(MuModel *model, ReplayBuffer *rb,
                              const TrainerConfig *cfg);

#ifdef __cplusplus
}
#endif

#endif // MUZE_TRAINER_H
