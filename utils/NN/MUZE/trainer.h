#include "muzero_model.h"

typedef struct TrainerConfig {
  int batch_size;
  int train_steps;     // how many SGD steps per "train call"
  int min_replay_size; // warmup
  float lr;
} TrainerConfig;

void trainer_train_from_replay(MuModel *model, ReplayBuffer *rb,
                               const TrainerConfig *cfg);
