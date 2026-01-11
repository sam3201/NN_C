#ifndef MUZERO_MODEL_H
#define MUZERO_MODEL_H

#include <stddef.h>


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int obs_dim;
  int latent_dim;
  int action_count;
} MuConfig;

/* forward declare */
typedef struct TrainerConfig TrainerConfig;
typedef struct NN_t NN_t;
typedef struct MuModel MuModel;

typedef struct {
  int opt_repr;
  int opt_dyn;
  int opt_pred;
  int opt_vprefix;
  int opt_reward;

  int loss_repr;
  int loss_dyn;
  int loss_pred;
  int loss_vprefix;
  int loss_reward;

  int lossd_repr;
  int lossd_dyn;
  int lossd_pred;
  int lossd_vprefix;
  int lossd_reward;

  long double lr_repr;
  long double lr_dyn;
  long double lr_pred;
  long double lr_vprefix;
  long double lr_reward;

  long double lr_mult_repr_start;
  long double lr_mult_repr_end;
  size_t lr_mult_repr_steps;
  long double lr_mult_dyn_start;
  long double lr_mult_dyn_end;
  size_t lr_mult_dyn_steps;
  long double lr_mult_pred_start;
  long double lr_mult_pred_end;
  size_t lr_mult_pred_steps;
  long double lr_mult_vprefix_start;
  long double lr_mult_vprefix_end;
  size_t lr_mult_vprefix_steps;
  long double lr_mult_reward_start;
  long double lr_mult_reward_end;
  size_t lr_mult_reward_steps;

  size_t hidden_repr;
  size_t hidden_dyn;
  size_t hidden_pred;
  size_t hidden_vprefix;
  size_t hidden_reward;

  int use_value_support;
  int use_reward_support;
  int support_size;
  float support_min;
  float support_max;

  int action_embed_dim;

  float w_policy;
  float w_value;
  float w_vprefix;
  float w_latent;
  float w_reward;

  float grad_clip;
  float global_grad_clip;
} MuNNConfig;

struct MuModel {
  MuConfig cfg;

  /* optional learned weights (for the generic placeholder model) */
  float *repr_W;
  float *dyn_W;
  float *pred_W;
  int repr_W_count;
  int dyn_W_count;
  int pred_W_count;

  float *rew_W; // [latent_dim]
  float rew_b;  // scalar bias
  int rew_W_count;

  void *runtime;

  /* dispatch hooks (MuZero style) */
  void (*repr)(MuModel *, const float *, float *);
  void (*predict)(MuModel *, const float *, float *, float *);
  void (*dynamics)(MuModel *, const float *, int, float *, float *);
  void (*train_policy_value)(MuModel *, const float *, const float *,
                             const float *, const float *, int, float);
  void (*train_dynamics)(MuModel *, const float *, const int *, const float *,
                         const float *, const float *, int, int, float,
                         float *out_latent_mse, float *out_reward_mse);
  void (*train_unroll)(MuModel *, const float *, const float *, const float *,
                       const float *, const int *, const float *,
                       const int *, const float *, int, int, int, float,
                       float, float *out_policy_loss, float *out_value_loss,
                       float *out_reward_loss, float *out_latent_loss);

  float *vprefix_W; // [latent_dim]
  float vprefix_b;  // scalar bias
  int vprefix_W_count;

  int value_norm_enabled;
  float value_min;
  float value_max;
  float value_rescale_eps;

  int use_nn;
  NN_t *nn_repr;
  NN_t *nn_dyn;
  NN_t *nn_pred;
  NN_t *nn_vprefix;
  NN_t *nn_reward;

  int use_value_support;
  int use_reward_support;
  int support_size;
  float support_min;
  float support_max;

  int action_embed_dim;
  int action_embed_count;
  float *action_embed;

  float w_policy;
  float w_value;
  float w_vprefix;
  float w_latent;
  float w_reward;

  float grad_clip;
};

/* creation / destruction */
MuModel *mu_model_create(const MuConfig *cfg);
MuModel *mu_model_create_nn(const MuConfig *cfg);
MuModel *mu_model_create_nn_with_cfg(const MuConfig *cfg,
                                     const MuNNConfig *nn_cfg);
void mu_model_free(MuModel *m);
void mu_model_copy_weights(MuModel *dst, const MuModel *src);

/* wrappers used by MCTS (these call m->repr/predict/dynamics if set) */
void mu_model_repr(MuModel *m, const float *obs, float *latent_out);
void mu_model_dynamics(MuModel *m, const float *latent_in, int action,
                       float *latent_out, float *reward_out);
void mu_model_predict(MuModel *m, const float *latent_in,
                      float *policy_logits_out, float *value_out);
float mu_model_denorm_value(MuModel *m, float v_norm);
float mu_model_value_transform(MuModel *m, float v);
float mu_model_value_transform_inv(MuModel *m, float v_norm);
int mu_model_predict_value_support(MuModel *m, const float *latent,
                                   float *out_probs, int max_bins);
int mu_model_predict_reward_support(MuModel *m, const float *latent,
                                    float *out_probs, int max_bins);
float mu_model_support_expected(MuModel *m, const float *probs, int bins);

/* ---- batch helpers used by trainer.c ---- */
int muzero_model_obs_dim(MuModel *m);
int muzero_model_action_count(MuModel *m);

void muzero_model_forward_batch(MuModel *m, const float *obs_batch, int B,
                                float *p_out, float *v_out);

void muzero_model_train_batch(MuModel *m, const float *obs_batch,
                              const float *pi_batch, const float *z_batch,
                              const float *weights, int B, float lr);
void muzero_model_train_dynamics_batch(
    MuModel *m, const float *obs_batch, const int *a_batch,
    const float *r_batch, const float *next_obs_batch, const float *weights,
    int train_reward_head, int B, float lr,
    float *out_latent_mse, // optional (can be NULL)
    float *out_reward_mse  // optional (can be NULL)
);

void muzero_model_train_unroll_batch(
    MuModel *m, const float *obs_seq, const float *pi_seq,
    const float *z_seq, const float *vprefix_seq, const int *a_seq,
    const float *r_seq, const int *done_seq, int B, int unroll_steps,
    int bootstrap_steps, float discount, float lr,
    const float *weights, float *out_policy_loss, float *out_value_loss,
    float *out_reward_loss, float *out_latent_loss);
void mu_model_step(MuModel *m, const float *obs, int action, float reward);
void mu_model_end_episode(MuModel *m, float terminal_reward);
void mu_model_reset_episode(MuModel *m);
void mu_model_train(MuModel *m);
void mu_model_train_with_cfg(MuModel *m, const TrainerConfig *cfg); // NEW

int mu_model_save(MuModel *m, const char *filename);
MuModel *mu_model_load(const char *filename);

/* ---- toy model (MuZero-style “real model” for the toy env) ---- */
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
