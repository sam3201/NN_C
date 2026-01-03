#ifndef MUZERO_MODEL_H
#define MUZERO_MODEL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------
   MuZero configuration
   ------------------------ */
typedef struct {
  int obs_dim;      // dimension of input observation
  int latent_dim;   // learned hidden state size
  int action_count; // number of discrete actions
} MuConfig;

/* ------------------------
   MuZero model structure
   ------------------------ */
typedef struct {
  MuConfig cfg;

  /* Weights for the three MuZero networks:
     - representation f(obs) -> latent
     - dynamics g(latent, action) -> latent+reward
     - prediction h(latent) -> policy, value

     NOTE: These are placeholders. Later you will replace them
     with NN_C layers or parameter buffers.
  */
  float *repr_W;
  float *dyn_W;
  float *pred_W;

  int repr_W_count;
  int dyn_W_count;
  int pred_W_count;

} MuModel;

/* ------------------------
   Constructor / Destructor
   ------------------------ */
MuModel *mu_model_create(const MuConfig *cfg);
void mu_model_free(MuModel *m);

/* ------------------------
   MuZero core functions
   ------------------------ */
void mu_model_repr(MuModel *m, const float *obs, float *latent_out);
void mu_model_dynamics(MuModel *m, const float *latent_in, int action,
                       float *latent_out, float *reward_out);
void mu_model_predict(MuModel *m, const float *latent_in,
                      float *policy_logits_out, float *value_out);

#ifdef __cplusplus
}
#endif
#endif
