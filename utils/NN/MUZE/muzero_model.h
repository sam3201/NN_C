#ifndef MUZERO_MODEL_H
#define MUZERO_MODEL_H

typedef struct {
    int action_count;
    int latent_dim;
    int state_dim;
    int reward_support_size;
} MuConfig;

typedef struct {
    MuConfig cfg;

    float *repr_weights;
    float *dyn_weights;
    float *pred_weights;

    int repr_weight_count;
    int dyn_weight_count;
    int pred_weight_count;
} MuModel;

MuModel *mu_create(MuConfig cfg);
void mu_free(MuModel *model);

// Core functions used by MCTS
void mu_initial_inference(
    MuModel *model,
    float *observation,
    float *latent_out,
    float *policy_logits_out,
    float *value_out
);

void mu_recurrent_inference(
    MuModel *model,
    float *latent_in,
    int action,
    float *latent_out,
    float *policy_logits_out,
    float *value_out,
    float *reward_out
);

#endif

