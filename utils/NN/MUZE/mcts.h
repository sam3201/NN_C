#ifndef MCTS_H
#define MCTS_H

#include "muzero_model.h"

#ifdef __cplusplus
extern "C" {
#endif

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
  float *root_value_dist; // support distribution, caller frees
  int root_value_bins;
} MCTSResult;

typedef struct {
  void *ctx;
  // returns uniform float in [0,1)
  float (*rand01)(void *ctx);
} MCTSRng;

MCTSResult mcts_run(MuModel *model, const float *obs, const MCTSParams *params,
                    MCTSRng *rng);
MCTSResult mcts_run_latent(MuModel *model, const float *latent,
                           const MCTSParams *params, MCTSRng *rng);

void mcts_result_free(MCTSResult *res);

#ifdef __cplusplus
}
#endif
#endif
