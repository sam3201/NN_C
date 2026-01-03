#ifndef MCTS_H
#define MCTS_H

#include "muzero_model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int num_simulations;
  float c_puct; /* PUCT constant */
  int max_depth;
  /* Root exploration noise */
  float dirichlet_alpha; /* if >0, add Dirichlet noise to root priors */
  float dirichlet_eps;   /* mixing factor for Dirichlet noise at root [0..1] */
  /* temperature for sampling action from visit counts (during self-play) */
  float temperature;
  /* discount factor for multi-step returns inside MCTS backups */
  float discount;
} MCTSParams;

typedef struct {
  int chosen_action;
  int action_count;
  float
      *pi; /* normalized visit-count policy (len action_count) - caller frees */
  float root_value; /* estimated value of root after search */
} MCTSResult;

/* Run MCTS using the muzero model. obs length must be model->cfg.obs_dim.
   Returns MCTSResult; caller should free reuslt via mcts_result_free(&res)  */
MCTSResult mcts_run(MuModel *model, const float *obs, const MCTSParams *params);

/* Free result resources */
void mcts_result_free(MCTSResult *res);

#ifdef __cplusplus
}
#endif

#endif /* MCTS_H */
