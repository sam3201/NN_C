#include "policy_value_network.h"
#include "../encoder/observation_encoder.h"
#include <string.h>
#include <time.h>

// Helper function to sample from categorical distribution
int sample_categorical(float* logits, int num_categories) {
    // Apply softmax to get probabilities
    float max_logit = -INFINITY;
    for (int i = 0; i < num_categories; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    float sum_exp = 0.0f;
    float probs[num_categories];
    for (int i = 0; i < num_categories; i++) {
        probs[i] = expf(logits[i] - max_logit);
        sum_exp += probs[i];
    }
    
    for (int i = 0; i < num_categories; i++) {
        probs[i] /= sum_exp;
    }
    
    // Sample using cumulative distribution
    float cumsum = 0.0f;
    float rand_val = (float)rand() / RAND_MAX;
    for (int i = 0; i < num_categories; i++) {
        cumsum += probs[i];
        if (rand_val < cumsum) {
            return i;
        }
    }
    
    return num_categories - 1; // Fallback
}

// Helper function to sample from Gaussian distribution
void sample_gaussian(float mean, float std, float* sample) {
    // Box-Muller transform for Gaussian sampling
    static int has_spare = 0;
    static float spare;
    
    if (has_spare) {
        *sample = spare * std + mean;
        has_spare = 0;
        return;
    }
    
    has_spare = 1;
    
    float u, v, s;
    do {
        u = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        v = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);
    
    s = sqrtf(-2.0f * logf(s) / s);
    spare = v * s;
    *sample = u * s * std + mean;
}

// Calculate log probability for categorical distribution
float categorical_log_prob(float* logits, int category, int num_categories) {
    float max_logit = -INFINITY;
    for (int i = 0; i < num_categories; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < num_categories; i++) {
        sum_exp += expf(logits[i] - max_logit);
    }
    
    return logits[category] - max_logit - logf(sum_exp);
}

// Calculate log probability for Gaussian distribution
float gaussian_log_prob(float sample, float mean, float std) {
    float diff = sample - mean;
    return -0.5f * logf(2.0f * M_PI * std * std) - (diff * diff) / (2.0f * std * std);
}

// Create policy and value network
PolicyValueNetwork* policy_value_network_create(RecurrentType recurrent_type) {
    PolicyValueNetwork* network = malloc(sizeof(PolicyValueNetwork));
    if (!network) return NULL;
    
    network->recurrent_type = recurrent_type;
    network->training_mode = 1;
    
    // Create shared core network
    size_t core_layers[] = {LATENT_DIM, LATENT_DIM * 2, LATENT_DIM * 2, LATENT_DIM};
    ActivationFunctionType core_actFuncs[] = {RELU, RELU, RELU, LINEAR};
    ActivationDerivativeType core_actDerivs[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, RELU_DERIVATIVE, LINEAR_DERIVATIVE};
    network->shared_core = NN_init(core_layers, core_actFuncs, core_actDerivs, MSE, MSE_DERIVATIVE, L1, SGD, 0.001f);
    if (!network->shared_core) {
        free(network);
        return NULL;
    }
    
    // Create recurrent component (simplified LSTM-like)
    network->hidden_state_size = LATENT_DIM * 2;
    network->hidden_state = malloc(sizeof(float) * network->hidden_state_size);
    if (!network->hidden_state) {
        NN_destroy(network->shared_core);
        free(network);
        return NULL;
    }
    
    // Initialize hidden state
    for (int i = 0; i < network->hidden_state_size; i++) {
        network->hidden_state[i] = 0.0f;
    }
    
    // Create movement head
    size_t movement_layers[] = {LATENT_DIM, NUM_DISCRETE_MOVEMENTS};
    ActivationFunctionType movement_actFuncs[] = {LINEAR};
    ActivationDerivativeType movement_actDerivs[] = {LINEAR_DERIVATIVE};
    network->movement_head = NN_init(movement_layers, movement_actFuncs, movement_actDerivs, MSE, MSE_DERIVATIVE, L1, SGD, 0.001f);
    
    // Create action head
    size_t action_layers[] = {LATENT_DIM, NUM_DISCRETE_ACTIONS};
    ActivationFunctionType action_actFuncs[] = {LINEAR};
    ActivationDerivativeType action_actDerivs[] = {LINEAR_DERIVATIVE};
    network->action_head = NN_init(action_layers, action_actFuncs, action_actDerivs, MSE, MSE_DERIVATIVE, L1, SGD, 0.001f);
    
    // Create aim heads
    size_t aim_mean_layers[] = {LATENT_DIM, 2};
    ActivationFunctionType aim_mean_actFuncs[] = {LINEAR};
    ActivationDerivativeType aim_mean_actDerivs[] = {LINEAR_DERIVATIVE};
    network->aim_mean_head = NN_init(aim_mean_layers, aim_mean_actFuncs, aim_mean_actDerivs, MSE, MSE_DERIVATIVE, L1, SGD, 0.001f);
    
    size_t aim_std_layers[] = {LATENT_DIM, 2};
    ActivationFunctionType aim_std_actFuncs[] = {LINEAR};
    ActivationDerivativeType aim_std_actDerivs[] = {LINEAR_DERIVATIVE};
    network->aim_std_head = NN_init(aim_std_layers, aim_std_actFuncs, aim_std_actDerivs, MSE, MSE_DERIVATIVE, L1, SGD, 0.001f);
    
    // Create charge heads
    size_t charge_mean_layers[] = {LATENT_DIM, 1};
    ActivationFunctionType charge_mean_actFuncs[] = {LINEAR};
    ActivationDerivativeType charge_mean_actDerivs[] = {LINEAR_DERIVATIVE};
    network->charge_mean_head = NN_init(charge_mean_layers, charge_mean_actFuncs, charge_mean_actDerivs, MSE, MSE_DERIVATIVE, L1, SGD, 0.001f);
    
    size_t charge_std_layers[] = {LATENT_DIM, 1};
    ActivationFunctionType charge_std_actFuncs[] = {LINEAR};
    ActivationDerivativeType charge_std_actDerivs[] = {LINEAR_DERIVATIVE};
    network->charge_std_head = NN_init(charge_std_layers, charge_std_actFuncs, charge_std_actDerivs, MSE, MSE_DERIVATIVE, L1, SGD, 0.001f);
    
    // Create value head
    size_t value_layers[] = {LATENT_DIM, 1};
    ActivationFunctionType value_actFuncs[] = {LINEAR};
    ActivationDerivativeType value_actDerivs[] = {LINEAR_DERIVATIVE};
    network->value_head = NN_init(value_layers, value_actFuncs, value_actDerivs, MSE, MSE_DERIVATIVE, L1, SGD, 0.001f);
    
    // Check all heads were created successfully
    if (!network->movement_head || !network->action_head || !network->aim_mean_head || 
        !network->aim_std_head || !network->charge_mean_head || !network->charge_std_head || 
        !network->value_head) {
        // Cleanup on failure
        if (network->movement_head) NN_destroy(network->movement_head);
        if (network->action_head) NN_destroy(network->action_head);
        if (network->aim_mean_head) NN_destroy(network->aim_mean_head);
        if (network->aim_std_head) NN_destroy(network->aim_std_head);
        if (network->charge_mean_head) NN_destroy(network->charge_mean_head);
        if (network->charge_std_head) NN_destroy(network->charge_std_head);
        if (network->value_head) NN_destroy(network->value_head);
        NN_destroy(network->shared_core);
        free(network->hidden_state);
        free(network);
        return NULL;
    }
    
    // Initialize current latent
    memset(network->current_latent, 0, sizeof(float) * LATENT_DIM);
    
    return network;
}

// Forward pass through policy and value network
PolicyOutput policy_value_network_forward(PolicyValueNetwork* network, float* latent_vector) {
    PolicyOutput output;
    
    // Store current latent
    for (int i = 0; i < LATENT_DIM; i++) {
        network->current_latent[i] = latent_vector[i];
    }
    
    // Simple recurrent update (concatenate current latent with previous hidden state)
    float recurrent_input[LATENT_DIM + network->hidden_state_size];
    for (int i = 0; i < LATENT_DIM; i++) {
        recurrent_input[i] = latent_vector[i];
    }
    for (int i = 0; i < network->hidden_state_size; i++) {
        recurrent_input[LATENT_DIM + i] = network->hidden_state[i];
    }
    
    // Pass through shared core
    long double* core_input = malloc(sizeof(long double) * (LATENT_DIM + network->hidden_state_size));
    for (int i = 0; i < LATENT_DIM + network->hidden_state_size; i++) {
        core_input[i] = (long double)recurrent_input[i];
    }
    
    long double* core_output = NN_forward(network->shared_core, core_input);
    
    // Update hidden state (simple LSTM-like update)
    for (int i = 0; i < network->hidden_state_size; i++) {
        network->hidden_state[i] = (float)core_output[i];
    }
    
    // Use core output for policy and value heads
    float core_features[LATENT_DIM];
    for (int i = 0; i < LATENT_DIM; i++) {
        core_features[i] = (float)core_output[i];
    }
    
    // Movement head
    long double* movement_input = malloc(sizeof(long double) * LATENT_DIM);
    for (int i = 0; i < LATENT_DIM; i++) {
        movement_input[i] = (long double)core_features[i];
    }
    long double* movement_logits = NN_forward(network->movement_head, movement_input);
    
    // Action head
    long double* action_logits = NN_forward(network->action_head, movement_input);
    
    // Aim heads
    long double* aim_mean = NN_forward(network->aim_mean_head, movement_input);
    long double* aim_std = NN_forward(network->aim_std_head, movement_input);
    
    // Apply softplus to ensure positive std
    for (int i = 0; i < 2; i++) {
        aim_std[i] = logf(1.0f + expf((float)aim_std[i]));
    }
    
    // Charge heads
    long double* charge_mean = NN_forward(network->charge_mean_head, movement_input);
    long double* charge_std = NN_forward(network->charge_std_head, movement_input);
    charge_std[0] = logf(1.0f + expf((float)charge_std[0]));
    
    // Value head
    long double* value = NN_forward(network->value_head, movement_input);
    
    // Sample actions if in training mode
    if (network->training_mode) {
        output.sampled_movement = sample_categorical((float*)movement_logits, NUM_DISCRETE_MOVEMENTS);
        output.sampled_action = sample_categorical((float*)action_logits, NUM_DISCRETE_ACTIONS);
        
        sample_gaussian((float)aim_mean[0], (float)aim_std[0], &output.sampled_aim[0]);
        sample_gaussian((float)aim_mean[1], (float)aim_std[1], &output.sampled_aim[1]);
        sample_gaussian((float)charge_mean[0], (float)charge_std[0], &output.sampled_charge);
        
        // Calculate log probabilities
        output.movement_log_prob = categorical_log_prob((float*)movement_logits, output.sampled_movement, NUM_DISCRETE_MOVEMENTS);
        output.action_log_prob = categorical_log_prob((float*)action_logits, output.sampled_action, NUM_DISCRETE_ACTIONS);
        output.aim_log_prob = gaussian_log_prob(output.sampled_aim[0], (float)aim_mean[0], (float)aim_std[0]) + 
                           gaussian_log_prob(output.sampled_aim[1], (float)aim_mean[1], (float)aim_std[1]);
        output.charge_log_prob = gaussian_log_prob(output.sampled_charge, (float)charge_mean[0], (float)charge_std[0]);
    } else {
        // In inference mode, use mean values for continuous actions
        output.sampled_movement = 0; // Default to idle
        output.sampled_action = 0;
        output.sampled_aim[0] = (float)aim_mean[0];
        output.sampled_aim[1] = (float)aim_mean[1];
        output.sampled_charge = (float)charge_mean[0];
        
        output.movement_log_prob = 0.0f;
        output.action_log_prob = 0.0f;
        output.aim_log_prob = 0.0f;
        output.charge_log_prob = 0.0f;
    }
    
    // Store outputs
    for (int i = 0; i < NUM_DISCRETE_MOVEMENTS; i++) {
        output.movement_logits[i] = (float)movement_logits[i];
    }
    for (int i = 0; i < NUM_DISCRETE_ACTIONS; i++) {
        output.action_logits[i] = (float)action_logits[i];
    }
    output.aim_mean[0] = (float)aim_mean[0];
    output.aim_mean[1] = (float)aim_mean[1];
    output.aim_std[0] = (float)aim_std[0];
    output.aim_std[1] = (float)aim_std[1];
    output.charge_mean = (float)charge_mean[0];
    output.charge_std = (float)charge_std[0];
    
    // Cleanup
    free(core_input);
    free(movement_input);
    
    return output;
}

// Get state value
float policy_value_network_get_value(PolicyValueNetwork* network) {
    if (!network) return 0.0f;
    
    float core_features[LATENT_DIM];
    for (int i = 0; i < LATENT_DIM; i++) {
        core_features[i] = network->current_latent[i];
    }
    
    long double* input = malloc(sizeof(long double) * LATENT_DIM);
    for (int i = 0; i < LATENT_DIM; i++) {
        input[i] = (long double)core_features[i];
    }
    
    long double* value = NN_forward(network->value_head, input);
    float result = (float)value[0];
    
    free(input);
    return result;
}

// Reset network state
void policy_value_network_reset(PolicyValueNetwork* network) {
    if (!network) return;
    
    for (int i = 0; i < network->hidden_state_size; i++) {
        network->hidden_state[i] = 0.0f;
    }
    
    memset(network->current_latent, 0, sizeof(float) * LATENT_DIM);
}

// Set training mode
void policy_value_network_set_training_mode(PolicyValueNetwork* network, int training_mode) {
    if (network) {
        network->training_mode = training_mode;
    }
}

// Destroy policy and value network
void policy_value_network_destroy(PolicyValueNetwork* network) {
    if (!network) return;
    
    if (network->shared_core) NN_destroy(network->shared_core);
    if (network->movement_head) NN_destroy(network->movement_head);
    if (network->action_head) NN_destroy(network->action_head);
    if (network->aim_mean_head) NN_destroy(network->aim_mean_head);
    if (network->aim_std_head) NN_destroy(network->aim_std_head);
    if (network->charge_mean_head) NN_destroy(network->charge_mean_head);
    if (network->charge_std_head) NN_destroy(network->charge_std_head);
    if (network->value_head) NN_destroy(network->value_head);
    
    if (network->hidden_state) free(network->hidden_state);
    
    free(network);
}
