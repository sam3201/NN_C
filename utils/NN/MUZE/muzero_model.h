#ifndef MUZERO_MODEL_H
#define MUZERO_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MuModel MuModel;

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
  void (*repr)(MuModel *, const float *, float *);
  void (*predict)(MuModel *, const float *, float *, float *);
  void (*dynamics)(MuModel *, const float *, int, float *, float *);

} MuModel_t;

MuModel *mu_model_create(const MuConfig *cfg);
void mu_model_free(MuModel *m);

void mu_model_repr(MuModel *m, const float *obs, float *latent_out);
void mu_model_dynamics(MuModel *m, const float *latent_in, int action,
                       float *latent_out, float *reward_out);
void mu_model_predict(MuModel *m, const float *latent_in,
                      float *policy_logits_out, float *value_out);

// ---- batch helpers used by trainer.c ----
int muzero_model_obs_dim(MuModel *m);
int muzero_model_action_count(MuModel *m);

/* forward for a batch:
   obs_batch: [B * obs_dim]
   p_out:     [B * action_count]  (probabilities, softmaxed)
   v_out:     [B]                 (tanh value)
*/
void muzero_model_forward_batch(MuModel *m, const float *obs_batch, int B,
                                float *p_out, float *v_out);

/* one SGD step on (obs, pi, z) targets
   pi_batch: [B * action_count] (target distribution)
   z_batch:  [B]                (target value)
*/
void muzero_model_train_batch(MuModel *m, const float *obs_batch,
                              const float *pi_batch, const float *z_batch,
                              int B, float lr);
void mu_model_step(MuModel *m, const float *obs, int action, float reward);
void mu_model_end_episode(MuModel *m, float terminal_reward);
void mu_model_reset_episode(MuModel *m);
void mu_model_train(MuModel *m);
MuModel *mu_model_create_toy(int size, int action_count);
void mu_model_free_toy(MuModel *m);

void mu_model_repr_toy(MuModel *m, const float *obs, float *latent_out);
void mu_model_predict_toy(MuModel *m, const float *latent, float *policy_logits,
                          float *value_out);
void mu_model_dynamics_toy(MuModel *m, const float *latent, int action,
                           float *latent2_out, float *reward_out);

#ifdef __cplusplus
}
#endif
#endif
