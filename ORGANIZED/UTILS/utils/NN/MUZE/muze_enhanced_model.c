#include "muze_enhanced_model.h"
#include "discrete_actions.h"
#include "compressed_obs.h"
#include "muze_enhanced_config.h"
#include "../NN/NN.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Create enhanced MuZE model
MuzeEnhancedModel *muze_enhanced_model_create(MuzeEnhancedConfig *config) {
    if (!config) return NULL;
    
    // Validate configuration
    if (!validate_enhanced_config(config)) {
        printf("Error: Invalid enhanced configuration\n");
        return NULL;
    }
    
    MuzeEnhancedModel *model = malloc(sizeof(MuzeEnhancedModel));
    if (!model) return NULL;
    
    // Store configuration
    model->config = config;
    
    // Initialize model and NN configs
    model->model_config = malloc(sizeof(MuzeEnhancedModelConfig));
    model->nn_config = malloc(sizeof(MuzeEnhancedNNConfig));
    
    init_enhanced_model_config(model->model_config, config);
    init_enhanced_nn_config(model->nn_config, model->model_config);
    
    // Create compressed observation buffer
    model->compressed_obs_buffer = malloc(sizeof(CompressedObs));
    init_compressed_obs(model->compressed_obs_buffer);
    
    // Create encoder network (if using compressed obs)
    if (config->use_compressed_obs) {
        size_t encoder_layers[] = {model->nn_config->encoder_input_dim, 
                                 model->nn_config->encoder_hidden_dim, 
                                 model->nn_config->encoder_output_dim, 0};
        ActivationFunctionType encoder_activations[] = {RELU, RELU, RELU};
        ActivationDerivativeType encoder_derivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, RELU_DERIVATIVE};
        
        model->encoder = NN_init_with_weight_init(encoder_layers, encoder_activations, encoder_derivatives,
                                                  MSE, MSE_DERIVATIVE, L2, ADAM, 0.001L, HE);
        if (!model->encoder) {
            free(model->model_config);
            free(model->nn_config);
            free(model->compressed_obs_buffer);
            free(model);
            return NULL;
        }
    }
    
    // Create decoder network (if using world model)
    if (config->use_world_model) {
        size_t decoder_layers[] = {model->nn_config->decoder_input_dim, 
                                 model->nn_config->decoder_hidden_dim, 
                                 model->nn_config->decoder_output_dim, 0};
        ActivationFunctionType decoder_activations[] = {RELU, RELU, RELU};
        ActivationDerivativeType decoder_derivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, RELU_DERIVATIVE};
        
        model->decoder = NN_init_with_weight_init(decoder_layers, decoder_activations, decoder_derivatives,
                                                  MSE, MSE_DERIVATIVE, L2, ADAM, 0.001L, HE);
        if (!model->decoder) {
            muze_enhanced_model_destroy(model);
            return NULL;
        }
    }
    
    // Create actor network (if using continuous actor)
    if (config->use_continuous_actor) {
        size_t actor_layers[] = {model->nn_config->obs_dim, 
                                model->nn_config->actor_hidden_dim, 
                                model->nn_config->actor_output_dim, 0};
        ActivationFunctionType actor_activations[] = {RELU, RELU, RELU};
        ActivationDerivativeType actor_derivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, RELU_DERIVATIVE};
        
        model->actor = NN_init_with_weight_init(actor_layers, actor_activations, actor_derivatives,
                                              MSE, MSE_DERIVATIVE, L2, ADAM, 
                                              config->continuous_lr, HE);
        if (!model->actor) {
            muze_enhanced_model_destroy(model);
            return NULL;
        }
        
        // Create critic network
        size_t critic_layers[] = {model->nn_config->obs_dim, 
                                 model->nn_config->critic_hidden_dim, 
                                 model->nn_config->critic_output_dim, 0};
        ActivationFunctionType critic_activations[] = {RELU, RELU, RELU};
        ActivationDerivativeType critic_derivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, RELU_DERIVATIVE};
        
        model->critic = NN_init_with_weight_init(critic_layers, critic_activations, critic_derivatives,
                                               MSE, MSE_DERIVATIVE, L2, ADAM, 
                                               config->continuous_lr * 0.1, HE);
        if (!model->critic) {
            muze_enhanced_model_destroy(model);
            return NULL;
        }
    }
    
    // Create world model (if enabled)
    if (config->use_world_model) {
        size_t world_model_layers[] = {model->nn_config->obs_dim, 
                                    model->nn_config->world_model_hidden_dim, 
                                    model->nn_config->world_model_output_dim, 0};
        ActivationFunctionType world_activations[] = {RELU, RELU, RELU};
        ActivationDerivativeType world_derivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, RELU_DERIVATIVE};
        
        model->world_model = NN_init_with_weight_init(world_model_layers, world_activations, world_derivatives,
                                                   MSE, MSE_DERIVATIVE, L2, ADAM, 
                                                   config->world_model_lr, HE);
        if (!model->world_model) {
            muze_enhanced_model_destroy(model);
            return NULL;
        }
    }
    
    // Initialize training state
    model->training_step = 0;
    model->loss_value = 0.0f;
    model->use_continuous_actor = config ? config->use_continuous_actor : 0;
    
    return model;
}

// Destroy enhanced MuZE model
void muze_enhanced_model_destroy(MuzeEnhancedModel *model) {
    if (!model) return;
    
    if (model->encoder) NN_destroy(model->encoder);
    if (model->decoder) NN_destroy(model->decoder);
    if (model->actor) NN_destroy(model->actor);
    if (model->critic) NN_destroy(model->critic);
    if (model->world_model) NN_destroy(model->world_model);
    
    if (model->compressed_obs_buffer) {
        cleanup_compressed_obs(model->compressed_obs_buffer);
        free(model->compressed_obs_buffer);
    }
    
    if (model->model_config) free(model->model_config);
    if (model->nn_config) free(model->nn_config);
    
    free(model);
}

// Forward pass through enhanced model
void muze_enhanced_model_forward(MuzeEnhancedModel *model, CompressedObs *obs, 
                                int *action, float *value, float *policy) {
    if (!model || !obs || !action || !value || !policy) return;
    
    if (model->use_continuous_actor) {
        // Use actor-critic for continuous control
        long double *obs_array_ld = malloc(model->model_config->obs_dim * sizeof(long double));
        for (int i = 0; i < model->model_config->obs_dim; i++) {
            obs_array_ld[i] = (long double)((float*)obs)[i];
        }
        
        // Get action from policy network
        long double *policy_output = malloc(model->model_config->action_count * sizeof(long double));
        policy_output = NN_forward(model->actor, obs_array_ld);
        
        // Get value from critic network
        long double *value_output = malloc(sizeof(long double));
        value_output = NN_forward(model->critic, obs_array_ld);
        
        // Convert to float
        float *current_policy = malloc(model->model_config->action_count * sizeof(float));
        for (int i = 0; i < model->model_config->action_count; i++) {
            current_policy[i] = (float)policy_output[i];
        }
        float current_value = (float)*value_output;
        
        // Sample action from policy
        float policy_sum = 0.0f;
        for (int i = 0; i < model->model_config->action_count; i++) {
            policy_sum += current_policy[i];
        }
        
        float rand_val = (float)rand() / RAND_MAX;
        float cumulative = 0.0f;
        *action = 0;
        
        for (int i = 0; i < model->model_config->action_count; i++) {
            cumulative += current_policy[i] / policy_sum;
            if (rand_val < cumulative) {
                *action = i;
                break;
            }
        }
        
        *value = current_value;
        for (int i = 0; i < model->model_config->action_count; i++) {
            policy[i] = current_policy[i];
        }
        
        free(obs_array_ld);
        free(current_policy);
    } else {
        // Simple forward pass - set default values
        *action = 0;
        *value = 0.0f;
        for (int i = 0; i < 4; i++) {
            policy[i] = 0.25f;
        }
    }
}

// Training step for enhanced model
void muze_enhanced_model_train_step(MuzeEnhancedModel *model, CompressedObs *obs, 
                               int action, float reward, int done) {
    if (!model || !obs) return;
    
    if (model->use_continuous_actor) {
        // Actor-critic training
        float *obs_array = (float*)obs;
        
        // Get current policy and value
        float current_value = 0.0f;
        float *current_policy = malloc(model->model_config->action_count * sizeof(float));
        
        muze_enhanced_model_forward(model, obs, &action, &current_value, current_policy);
        
        // Compute TD error
        float next_value = reward + (done ? 0.0f : 0.99f * current_value);
        float td_error = next_value - current_value;
        
        // Update critic
        float *critic_loss = malloc(sizeof(float));
        *critic_loss = td_error * td_error;
        NN_backprop_custom_delta(model->critic, obs_array, (long double*)critic_loss);
        free(critic_loss);
        
        // Update actor
        float *actor_loss = malloc(sizeof(float));
        *actor_loss = -td_error * (current_policy[action] + 1e-8f);  // Add small epsilon for numerical stability
        NN_backprop_custom_delta(model->actor, obs_array, (long double*)actor_loss);
        free(actor_loss);
        
        // Update networks
        model->critic->optimizer(model->critic);
        model->actor->optimizer(model->actor);
        
        model->training_step++;
        model->loss_value = *critic_loss;
        
        free(current_policy);
    } else {
        // Use simplified training (no original MuZE)
        model->training_step++;
        model->loss_value = reward;
    }
    
    // Update world model (if enabled)
    if (model->world_model && model->world_model && 
        model->training_step % model->config->world_model_update_freq == 0) {
        
        float *obs_array = (float*)obs;
        float *latent = malloc(model->model_config->latent_dim * sizeof(float));
        
        // Encode observation to latent space
        long double *latent_ld = malloc(model->model_config->latent_dim * sizeof(long double));
        latent_ld = NN_forward(model->encoder, obs_array);
        
        // Convert to float
        for (int i = 0; i < model->model_config->latent_dim; i++) {
            latent[i] = (float)latent_ld[i];
        }
        
        // Decode back to observation space
        float *reconstructed = malloc(model->model_config->obs_dim * sizeof(float));
        long double *reconstructed_ld = malloc(model->model_config->obs_dim * sizeof(long double));
        reconstructed_ld = NN_forward(model->decoder, latent_ld);
        
        // Convert to float
        for (int i = 0; i < model->model_config->obs_dim; i++) {
            reconstructed[i] = (float)reconstructed_ld[i];
        }
        
        // Compute reconstruction loss
        float recon_loss = 0.0f;
        for (int i = 0; i < model->model_config->obs_dim; i++) {
            float diff = obs_array[i] - reconstructed[i];
            recon_loss += diff * diff;
        }
        recon_loss /= model->model_config->obs_dim;
        
        // Update world model
        float *world_loss = malloc(sizeof(float));
        *world_loss = recon_loss;
        NN_backprop_custom_delta(model->world_model, obs_array, (long double*)world_loss);
        free(world_loss);
        
        free(reconstructed_ld);
        free(reconstructed);
        free(latent_ld);
        free(latent);
        free(world_loss);
        
        model->world_model->optimizer(model->world_model);
    }
}

// Update model weights
void muze_enhanced_model_update_weights(MuzeEnhancedModel *model) {
    if (!model) return;
    
    if (model->encoder) model->encoder->optimizer(model->encoder);
    if (model->decoder) model->decoder->optimizer(model->decoder);
    if (model->actor) model->actor->optimizer(model->actor);
    if (model->critic) model->critic->optimizer(model->critic);
    if (model->world_model) model->world_model->optimizer(model->world_model);
}

// Encode multiple actions into single discrete action
int encode_action(float move, float forward, float turn, int attack, int harvest) {
    return pack_action(move, forward, turn, attack, harvest);
}

// Decode single discrete action into multiple actions
void decode_action(int action_id, float *move, float *forward, float *turn, int *attack, int *harvest) {
    unpack_action(action_id, move, forward, turn, attack, harvest);
}

// Print model summary
void muze_enhanced_model_print_summary(MuzeEnhancedModel *model) {
    if (!model) return;
    
    printf("=== Enhanced MuZE Model Summary ===\n");
    printf("Configuration:\n");
    printf("  Discrete Actions: %s\n", model->config->use_discrete_actions ? "Yes" : "No");
    printf("  Compressed Obs: %s\n", model->config->use_compressed_obs ? "Yes" : "No");
    printf("  Continuous Actor: %s\n", model->config->use_continuous_actor ? "Yes" : "No");
    printf("  World Model: %s\n", model->config->use_world_model ? "Yes" : "No");
    
    printf("\nModel Dimensions:\n");
    printf("  Obs Dim: %d\n", model->model_config->obs_dim);
    printf("  Latent Dim: %d\n", model->model_config->latent_dim);
    printf("  Action Count: %d\n", model->model_config->action_count);
    
    printf("\nNetworks:\n");
    if (model->base_model) printf("  Base Model: ✓\n");
    if (model->encoder) printf("  Encoder: ✓\n");
    if (model->decoder) printf("  Decoder: ✓\n");
    if (model->actor) printf("  Actor: ✓\n");
    if (model->critic) printf("  Critic: ✓\n");
    if (model->world_model) printf("  World Model: ✓\n");
    
    printf("\nTraining:\n");
    printf("  Training Step: %d\n", model->training_step);
    printf("  Loss Value: %.6f\n", model->loss_value);
    printf("========================\n");
}

// Get total parameter count
size_t muze_enhanced_model_get_parameter_count(MuzeEnhancedModel *model) {
    if (!model) return 0;
    
    // Return a placeholder count
    return 1000; // Placeholder
}

// Save model to file
int muze_enhanced_model_save(MuzeEnhancedModel *model, const char *filename) {
    if (!model || !filename) return 0;
    
    // Save configuration and state
    FILE *file = fopen(filename, "wb");
    if (!file) return 0;
    
    // Write basic model info
    fwrite(&model->training_step, sizeof(int), 1, file);
    fwrite(&model->loss_value, sizeof(float), 1, file);
    fwrite(&model->use_continuous_actor, sizeof(int), 1, file);
    
    fclose(file);
    return 1;
}

// Load model from file
MuzeEnhancedModel *muze_enhanced_model_load(const char *filename) {
    if (!filename) return NULL;
    
    FILE *file = fopen(filename, "rb");
    if (!file) return NULL;
    
    // Create a model with properly initialized config (not just {0})
    MuzeEnhancedConfig config;
    init_enhanced_config(&config);
    // Set continuous actor mode for CLI compatibility
    config.use_discrete_actions = 0;
    config.use_continuous_actor = 1;
    config.pack_actions = 0;
    
    MuzeEnhancedModel *model = muze_enhanced_model_create(&config);
    if (!model) {
        fclose(file);
        return NULL;
    }
    
    // Read basic model info
    fread(&model->training_step, sizeof(int), 1, file);
    fread(&model->loss_value, sizeof(float), 1, file);
    fread(&model->use_continuous_actor, sizeof(int), 1, file);
    
    fclose(file);
    return model;
}
