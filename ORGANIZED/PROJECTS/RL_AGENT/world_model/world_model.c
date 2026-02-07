#include "world_model.h"
#include "../policy/policy_value_network.h"
#include <string.h>

// Helper function to initialize RSSM state
void init_rssm_state(RSSMState* state) {
    if (!state) return;
    
    memset(state->deterministic_state, 0, sizeof(float) * WORLD_MODEL_LATENT_DIM);
    memset(state->stochastic_state, 0, sizeof(float) * WORLD_MODEL_LATENT_DIM);
    memset(state->posterior_state, 0, sizeof(float) * WORLD_MODEL_LATENT_DIM);
}

// Create world model
WorldModel* world_model_create(int use_transformer) {
    WorldModel* model = malloc(sizeof(WorldModel));
    if (!model) return NULL;
    
    model->use_transformer = use_transformer;
    model->learning_rate = 0.001f;
    
    // Initialize RSSM state
    init_rssm_state(&model->current_state);
    
    // Create transition model (predicts next state from current state and action)
    size_t transition_layers[] = {WORLD_MODEL_LATENT_DIM + WORLD_MODEL_ACTION_DIM, 
                               WORLD_MODEL_LATENT_DIM * 2, 
                               WORLD_MODEL_LATENT_DIM * 2, 
                               WORLD_MODEL_LATENT_DIM * 2};
    ActivationFunctionType transition_actFuncs[] = {RELU, RELU, RELU, LINEAR};
    ActivationDerivativeType transition_actDerivs[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, RELU_DERIVATIVE, LINEAR_DERIVATIVE};
    model->transition_model = NN_init(transition_layers, transition_actFuncs, transition_actDerivs, MSE, MSE_DERIVATIVE, L1, SGD, model->learning_rate);
    if (!model->transition_model) {
        free(model);
        return NULL;
    }
    
    // Create reward predictor
    size_t reward_layers[] = {WORLD_MODEL_LATENT_DIM + WORLD_MODEL_ACTION_DIM, 
                           WORLD_MODEL_LATENT_DIM, 
                           WORLD_MODEL_LATENT_DIM / 2, 
                           1};
    ActivationFunctionType reward_actFuncs[] = {RELU, RELU, LINEAR};
    ActivationDerivativeType reward_actDerivs[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, LINEAR_DERIVATIVE};
    model->reward_predictor = NN_init(reward_layers, reward_actFuncs, reward_actDerivs, MSE, MSE_DERIVATIVE, L1, SGD, model->learning_rate);
    if (!model->reward_predictor) {
        NN_destroy(model->transition_model);
        free(model);
        return NULL;
    }
    
    // Create value predictor (optional)
    size_t value_layers[] = {WORLD_MODEL_LATENT_DIM, 
                          WORLD_MODEL_LATENT_DIM / 2, 
                          1};
    ActivationFunctionType value_actFuncs[] = {RELU, LINEAR};
    ActivationDerivativeType value_actDerivs[] = {RELU_DERIVATIVE, LINEAR_DERIVATIVE};
    model->value_predictor = NN_init(value_layers, value_actFuncs, value_actDerivs, MSE, MSE_DERIVATIVE, L1, SGD, model->learning_rate);
    if (!model->value_predictor) {
        NN_destroy(model->transition_model);
        NN_destroy(model->reward_predictor);
        free(model);
        return NULL;
    }
    
    // Create posterior model (updates state with new observations)
    size_t posterior_layers[] = {WORLD_MODEL_LATENT_DIM + WORLD_MODEL_LATENT_DIM + WORLD_MODEL_ACTION_DIM,
                               WORLD_MODEL_LATENT_DIM * 2,
                               WORLD_MODEL_LATENT_DIM * 2,
                               WORLD_MODEL_LATENT_DIM * 2};
    ActivationFunctionType posterior_actFuncs[] = {RELU, RELU, RELU, LINEAR};
    ActivationDerivativeType posterior_actDerivs[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, RELU_DERIVATIVE, LINEAR_DERIVATIVE};
    model->posterior_model = NN_init(posterior_layers, posterior_actFuncs, posterior_actDerivs, MSE, MSE_DERIVATIVE, L1, SGD, model->learning_rate);
    if (!model->posterior_model) {
        NN_destroy(model->transition_model);
        NN_destroy(model->reward_predictor);
        NN_destroy(model->value_predictor);
        free(model);
        return NULL;
    }
    
    // Create Transformer for sequence modeling (optional)
    if (use_transformer) {
        model->transformer_model = TRANSFORMER_init(WORLD_MODEL_LATENT_DIM, 4, 2);
        if (!model->transformer_model) {
            NN_destroy(model->transition_model);
            NN_destroy(model->reward_predictor);
            NN_destroy(model->value_predictor);
            NN_destroy(model->posterior_model);
            free(model);
            return NULL;
        }
    } else {
        model->transformer_model = NULL;
    }
    
    // Initialize loss tracking
    model->transition_loss = 0.0f;
    model->reward_loss = 0.0f;
    model->posterior_loss = 0.0f;
    
    // Initialize imagined trajectory buffer
    memset(model->imagined_states, 0, sizeof(float) * WORLD_MODEL_HORIZON * WORLD_MODEL_LATENT_DIM);
    memset(model->imagined_rewards, 0, sizeof(float) * WORLD_MODEL_HORIZON);
    memset(model->imagined_actions, 0, sizeof(float) * WORLD_MODEL_HORIZON * WORLD_MODEL_ACTION_DIM);
    
    return model;
}

// Reset world model state
void world_model_reset(WorldModel* model) {
    if (!model) return;
    
    init_rssm_state(&model->current_state);
    
    model->transition_loss = 0.0f;
    model->reward_loss = 0.0f;
    model->posterior_loss = 0.0f;
    
    memset(model->imagined_states, 0, sizeof(float) * WORLD_MODEL_HORIZON * WORLD_MODEL_LATENT_DIM);
    memset(model->imagined_rewards, 0, sizeof(float) * WORLD_MODEL_HORIZON);
    memset(model->imagined_actions, 0, sizeof(float) * WORLD_MODEL_HORIZON * WORLD_MODEL_ACTION_DIM);
}

// Update world model with new experience
void world_model_update(WorldModel* model, float* latent_state, float* action, float* next_latent, float reward) {
    if (!model || !latent_state || !action || !next_latent) return;
    
    // Prepare input for posterior model (current state + action + next observation)
    float posterior_input[WORLD_MODEL_LATENT_DIM + WORLD_MODEL_LATENT_DIM + WORLD_MODEL_ACTION_DIM];
    for (int i = 0; i < WORLD_MODEL_LATENT_DIM; i++) {
        posterior_input[i] = model->current_state.posterior_state[i];
        posterior_input[WORLD_MODEL_LATENT_DIM + i] = next_latent[i];
    }
    for (int i = 0; i < WORLD_MODEL_ACTION_DIM; i++) {
        posterior_input[2 * WORLD_MODEL_LATENT_DIM + i] = action[i];
    }
    
    // Update posterior state
    long double* posterior_input_ld = malloc(sizeof(long double) * (2 * WORLD_MODEL_LATENT_DIM + WORLD_MODEL_ACTION_DIM));
    for (int i = 0; i < 2 * WORLD_MODEL_LATENT_DIM + WORLD_MODEL_ACTION_DIM; i++) {
        posterior_input_ld[i] = (long double)posterior_input[i];
    }
    
    long double* posterior_output = NN_forward(model->posterior_model, posterior_input_ld);
    
    // Update current posterior state
    for (int i = 0; i < WORLD_MODEL_LATENT_DIM; i++) {
        model->current_state.posterior_state[i] = (float)posterior_output[i];
    }
    
    // Update deterministic state (simple LSTM-like update)
    for (int i = 0; i < WORLD_MODEL_LATENT_DIM; i++) {
        model->current_state.deterministic_state[i] = 0.9f * model->current_state.deterministic_state[i] + 
                                                   0.1f * model->current_state.posterior_state[i];
    }
    
    // Update stochastic state (add noise for exploration)
    for (int i = 0; i < WORLD_MODEL_LATENT_DIM; i++) {
        float noise = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        model->current_state.stochastic_state[i] = model->current_state.posterior_state[i] + noise;
    }
    
    free(posterior_input_ld);
    
    // Train transition model
    float transition_input[WORLD_MODEL_LATENT_DIM + WORLD_MODEL_ACTION_DIM];
    for (int i = 0; i < WORLD_MODEL_LATENT_DIM; i++) {
        transition_input[i] = model->current_state.posterior_state[i];
    }
    for (int i = 0; i < WORLD_MODEL_ACTION_DIM; i++) {
        transition_input[WORLD_MODEL_LATENT_DIM + i] = action[i];
    }
    
    long double* transition_input_ld = malloc(sizeof(long double) * (WORLD_MODEL_LATENT_DIM + WORLD_MODEL_ACTION_DIM));
    for (int i = 0; i < WORLD_MODEL_LATENT_DIM + WORLD_MODEL_ACTION_DIM; i++) {
        transition_input_ld[i] = (long double)transition_input[i];
    }
    
    long double* transition_output = NN_forward(model->transition_model, transition_input_ld);
    
    // Calculate transition loss (MSE between predicted and actual next state)
    float transition_error = 0.0f;
    for (int i = 0; i < WORLD_MODEL_LATENT_DIM; i++) {
        float error = (float)transition_output[i] - next_latent[i];
        transition_error += error * error;
    }
    model->transition_loss = transition_error / WORLD_MODEL_LATENT_DIM;
    
    // Train reward predictor
    long double* reward_input = malloc(sizeof(long double) * (WORLD_MODEL_LATENT_DIM + WORLD_MODEL_ACTION_DIM));
    for (int i = 0; i < WORLD_MODEL_LATENT_DIM; i++) {
        reward_input[i] = (long double)model->current_state.posterior_state[i];
    }
    for (int i = 0; i < WORLD_MODEL_ACTION_DIM; i++) {
        reward_input[WORLD_MODEL_LATENT_DIM + i] = (long double)action[i];
    }
    
    long double* reward_output = NN_forward(model->reward_predictor, reward_input);
    float predicted_reward = (float)reward_output[0];
    
    // Calculate reward loss (MSE)
    model->reward_loss = (predicted_reward - reward) * (predicted_reward - reward);
    
    // Calculate posterior loss (KL divergence approximation)
    float posterior_error = 0.0f;
    for (int i = 0; i < WORLD_MODEL_LATENT_DIM; i++) {
        float error = (float)posterior_output[i] - model->current_state.posterior_state[i];
        posterior_error += error * error;
    }
    model->posterior_loss = posterior_error / WORLD_MODEL_LATENT_DIM;
    
    free(transition_input_ld);
    free(reward_input);
}

// Predict next state and reward
void world_model_predict(WorldModel* model, float* latent_state, float* action, float* next_pred_state, float* pred_reward) {
    if (!model || !latent_state || !action || !next_pred_state || !pred_reward) return;
    
    // Prepare input for transition model
    float transition_input[WORLD_MODEL_LATENT_DIM + WORLD_MODEL_ACTION_DIM];
    for (int i = 0; i < WORLD_MODEL_LATENT_DIM; i++) {
        transition_input[i] = latent_state[i];
    }
    for (int i = 0; i < WORLD_MODEL_ACTION_DIM; i++) {
        transition_input[WORLD_MODEL_LATENT_DIM + i] = action[i];
    }
    
    long double* input_ld = malloc(sizeof(long double) * (WORLD_MODEL_LATENT_DIM + WORLD_MODEL_ACTION_DIM));
    for (int i = 0; i < WORLD_MODEL_LATENT_DIM + WORLD_MODEL_ACTION_DIM; i++) {
        input_ld[i] = (long double)transition_input[i];
    }
    
    // Predict next state
    long double* state_output = NN_forward(model->transition_model, input_ld);
    for (int i = 0; i < WORLD_MODEL_LATENT_DIM; i++) {
        next_pred_state[i] = (float)state_output[i];
    }
    
    // Predict reward
    long double* reward_output = NN_forward(model->reward_predictor, input_ld);
    *pred_reward = (float)reward_output[0];
    
    free(input_ld);
}

// Imagine trajectory using world model
void world_model_imagine_trajectory(WorldModel* model, float* initial_state, PolicyOutput* policy, int horizon) {
    if (!model || !initial_state || !policy || horizon <= 0 || horizon > WORLD_MODEL_HORIZON) return;
    
    // Start from initial state
    float current_state[WORLD_MODEL_LATENT_DIM];
    for (int i = 0; i < WORLD_MODEL_LATENT_DIM; i++) {
        current_state[i] = initial_state[i];
        model->imagined_states[0][i] = initial_state[i];
    }
    
    // Generate imagined trajectory
    for (int t = 0; t < horizon; t++) {
        // Sample action from policy (simplified - use mean values)
        float action[WORLD_MODEL_ACTION_DIM];
        
        // Discrete actions (use sampled values from policy)
        action[0] = (float)policy->sampled_movement;  // movement
        action[1] = (float)policy->sampled_action;     // action
        
        // Continuous actions (use mean values)
        action[2] = policy->sampled_aim[0];            // aim_x
        action[3] = policy->sampled_aim[1];            // aim_y
        action[4] = policy->sampled_charge;          // charge
        
        // Store action
        for (int i = 0; i < WORLD_MODEL_ACTION_DIM; i++) {
            model->imagined_actions[t][i] = action[i];
        }
        
        // Predict next state and reward
        float next_state[WORLD_MODEL_LATENT_DIM];
        float pred_reward;
        world_model_predict(model, current_state, action, next_state, &pred_reward);
        
        // Store imagined state and reward
        for (int i = 0; i < WORLD_MODEL_LATENT_DIM; i++) {
            model->imagined_states[t + 1][i] = next_state[i];
        }
        model->imagined_rewards[t] = pred_reward;
        
        // Update current state for next iteration
        for (int i = 0; i < WORLD_MODEL_LATENT_DIM; i++) {
            current_state[i] = next_state[i];
        }
    }
}

// Train world model on batch of experiences
void world_model_train(WorldModel* model, float* states, float* actions, float* next_states, float* rewards, int batch_size) {
    if (!model || !states || !actions || !next_states || !rewards || batch_size <= 0) return;
    
    float total_transition_loss = 0.0f;
    float total_reward_loss = 0.0f;
    float total_posterior_loss = 0.0f;
    
    for (int b = 0; b < batch_size; b++) {
        int state_offset = b * WORLD_MODEL_LATENT_DIM;
        int action_offset = b * WORLD_MODEL_ACTION_DIM;
        
        world_model_update(model, 
                         &states[state_offset], 
                         &actions[action_offset], 
                         &next_states[state_offset], 
                         rewards[b]);
        
        total_transition_loss += model->transition_loss;
        total_reward_loss += model->reward_loss;
        total_posterior_loss += model->posterior_loss;
    }
    
    // Update loss tracking
    model->transition_loss = total_transition_loss / batch_size;
    model->reward_loss = total_reward_loss / batch_size;
    model->posterior_loss = total_posterior_loss / batch_size;
}

// Get loss values
float world_model_get_transition_loss(WorldModel* model) {
    return model ? model->transition_loss : 0.0f;
}

float world_model_get_reward_loss(WorldModel* model) {
    return model ? model->reward_loss : 0.0f;
}

float world_model_get_posterior_loss(WorldModel* model) {
    return model ? model->posterior_loss : 0.0f;
}

// Destroy world model
void world_model_destroy(WorldModel* model) {
    if (!model) return;
    
    if (model->transition_model) NN_destroy(model->transition_model);
    if (model->reward_predictor) NN_destroy(model->reward_predictor);
    if (model->value_predictor) NN_destroy(model->value_predictor);
    if (model->posterior_model) NN_destroy(model->posterior_model);
    if (model->transformer_model) TRANSFORMER_destroy(model->transformer_model);
    
    free(model);
}
