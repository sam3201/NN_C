#ifndef MUZERO_MODEL_H
#define MUZERO_MODEL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Basic config for the MuZero model wrapper */
typedef struct {
    int obs_dim;        /* dimension of raw observation vector */
    int latent_dim;     /* dimension of learned latent state */
    int action_count;   /* number of discrete actions */
} MuConfig;

/* Opaque model handle */
typedef struct {
    MuConfig cfg;

    float *repr_weights;
    float *dyn_weights;
    float *pred_weights;

    int repr_weight_count;
    int dyn_weight_count;
    int pred_weight_count;

    // You can add more members as the model expands
} MuModel;

/* Create / free */
MuModel *mu_model_create(const MuConfig *cfg);
void mu_model_free(MuModel *m);

/* Representation: maps observation -> latent state (h_out must be latent_dim floats) */
void mu_model_repr(MuModel *m, const float *obs, float *h_out);

/* Dynamics: given latent h_in and action -> produce h_out and reward_out */
void mu_model_dynamics(MuModel *m, const float *h_in, int action, float *h_out, float *reward_out);

/* Prediction: given latent h -> policy logits (len action_count) and scalar value */
void mu_model_predict(MuModel *m, const float *h, float *policy_logits_out, float *value_out);

/* Optional: save / load model params (implement with your serialization) */
int mu_model_save(MuModel *m, const char *path);
int mu_model_load(MuModel *m, const char *path);

#ifdef __cplusplus
}
#endif

#endif /* MUZERO_MODEL_H */

