#ifndef WORLD_MODEL_H
#define WORLD_MODEL_H

#include "../encoder/observation_encoder.h"
#include "../policy/policy_value_network.h"

// World model configuration
#define WORLD_MODEL_LATENT_DIM 128
#define WORLD_MODEL_ACTION_DIM NUM_CONTINUOUS_ACTIONS + NUM_DISCRETE_ACTIONS
#define WORLD_MODEL_HORIZON 10
#define WORLD_MODEL_NUM_LAYERS 3

// RSSM (Recurrent State-Space Model) components
typedef struct {
    float deterministic_state[WORLD_MODEL_LATENT_DIM];
    float stochastic_state[WORLD_MODEL_LATENT_DIM];
    float posterior_state[WORLD_MODEL_LATENT_DIM];
} RSSMState;

// World model structure
typedef struct {
    // Transition model (predicts next state)
    NN_t* transition_model;
    
    // Reward predictor
    NN_t* reward_predictor;
    
    // Value predictor (optional)
    NN_t* value_predictor;
    
    // Posterior model (updates state with new observations)
    NN_t* posterior_model;
    
    // Transformer for sequence modeling (optional)
    Transformer_t* transformer_model;
    
    // Current RSSM state
    RSSMState current_state;
    
    // Training parameters
    float learning_rate;
    int use_transformer;
    
    // Imagined trajectory buffer
    float imagined_states[WORLD_MODEL_HORIZON][WORLD_MODEL_LATENT_DIM];
    float imagined_rewards[WORLD_MODEL_HORIZON];
    float imagined_actions[WORLD_MODEL_HORIZON][WORLD_MODEL_ACTION_DIM];
    
    // Loss tracking
    float transition_loss;
    float reward_loss;
    float posterior_loss;
} WorldModel;

// Function declarations
WorldModel* world_model_create(int use_transformer);
void world_model_destroy(WorldModel* model);
void world_model_reset(WorldModel* model);
void world_model_update(WorldModel* model, float* latent_state, float* action, float* next_latent, float reward);
void world_model_predict(WorldModel* model, float* latent_state, float* action, float* next_pred_state, float* pred_reward);
void world_model_imagine_trajectory(WorldModel* model, float* initial_state, PolicyOutput* policy, int horizon);
void world_model_train(WorldModel* model, float* states, float* actions, float* next_states, float* rewards, int batch_size);
float world_model_get_transition_loss(WorldModel* model);
float world_model_get_reward_loss(WorldModel* model);
float world_model_get_posterior_loss(WorldModel* model);

#endif // WORLD_MODEL_H
