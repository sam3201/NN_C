#ifndef MUZERO_MODEL_H
#define MUZERO_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int obs_dim;
  int latent_dim;
  int action_count;
} MuConfig;

typedef struct {
  MuConfig cfg;
  float *repr_W;
  float *dyn_W;
  float *pred_W;
  int repr_W_count;
  int dyn_W_count;
  int pred_W_count;

  void *runtime;
} MuModel;

MuModel *mu_model_create(const MuConfig *cfg);
void mu_model_free(MuModel *m);

void mu_model_repr(MuModel *m, const float *obs, float *latent_out);
void mu_model_dynamics(MuModel *m, const float *latent_in, int action,
                       float *latent_out, float *reward_out);
void mu_model_predict(MuModel *m, const float *latent_in,
                      float *policy_logits_out, float *value_out);

#ifdef __cplusplus
}
#endif
#endif
