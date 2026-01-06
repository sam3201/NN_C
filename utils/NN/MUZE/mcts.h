#ifndef MCTS_H
#define MCTS_H

#include "muzero_model.h"

typedef struct {
  int num_simulations;
  float c_puct;
  int max_depth;
  float dirichlet_alpha;
  float dirichlet_eps;
  float temperature;
  float discount;
} MCTSParams;

typedef struct {
  int chosen_action;
  int action_count;
  float *pi;        // visit-count policy, caller frees
  float root_value; // estimated root value
} MCTSResult;

MCTSResult mcts_run(MuModel *model, const float *obs, const MCTSParams *params);
MCTSResult mcts_run_latent(MuModel *model, const float *latent,
                           const MCTSParams *params);

void mcts_result_free(MCTSResult *res);

#endif
#endif
