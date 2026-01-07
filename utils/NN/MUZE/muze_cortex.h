#ifndef MUZE_CORTEX_H
#define MUZE_CORTEX_H

#include "mcts.h"
#include "muzero_model.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MuCortex {
  void *brain;

  void (*encode)(void *brain, float *obs, size_t obs_dim,
                 long double ***latent_seq, size_t *seq_len);

  void (*policy)(void *brain, long double **latent_seq, size_t seq_len,
                 float *action_probs, size_t action_count);

  void (*learn)(void *brain, const float *obs, size_t obs_dim, int action,
                float reward, int terminal);

  void (*plan)(void *brain, const float *obs, size_t obs_dim,
               size_t action_count);

  void (*free_latent_seq)(void *brain, long double **latent_seq,
                          size_t seq_len);
  bool use_mcts;          /* false = argmax(policy), true = MCTS */
  MuModel *mcts_model;    /* required if use_mcts=1 */
  MCTSParams mcts_params; /* params used by MCTS */

} MuCortex;

#ifdef __cplusplus
}
#endif

#endif // MUZE_CORTEX_H
