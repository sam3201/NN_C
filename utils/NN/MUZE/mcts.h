#ifndef MCTS_H
#define MCTS_H

#include "muzero_model.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* MCTS parameters */
typedef struct {
    int num_simulations;
    float c_puct;      /* PUCT constant */
} MCTSParams;

/* Result of running MCTS on a root observation */
typedef struct {
    int action_count;
    float *pi;      /* normalized visit-count policy (len action_count) - caller frees */
    int chosen_action;
} MCTSResult;

/* Run MCTS using muzero model. obs is length obs_dim.
   Caller must free returned MCTSResult.pi */
MCTSResult mcts_run(MuModel *model, const float *obs, const MCTSParams *params);

void mcts_result_free(MCTSResult *res);

#ifdef __cplusplus
}
#endif

#endif /* MCTS_H */

