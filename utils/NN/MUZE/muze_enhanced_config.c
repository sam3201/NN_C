#include "muze_enhanced_config.h"
#include <string.h>

// Initialize enhanced configuration with sensible defaults
void init_enhanced_config(MuzeEnhancedConfig *config) {
    if (!config) return;
    
    // Initialize with zeros
    memset(config, 0, sizeof(MuzeEnhancedConfig));
    
    // Set enhanced action space defaults
    config->use_discrete_actions = 1;
    config->num_move_bins = 31;        // -1.0 to 1.0
    config->num_forward_bins = 11;      // 0.0 to 1.0
    config->num_turn_bins = 31;        // -1.0 to 1.0
    
    // Set compressed observation defaults
    config->use_compressed_obs = 1;
    config->num_rays = 128;
    config->voxel_grid_size = 16;      // 16x16x8
    config->local_grid_size = 8;        // 8x8
    config->max_token_value = 7;
    
    // Set hybrid configuration defaults (disabled by default)
    config->use_continuous_actor = 0;
    config->continuous_lr = 0.0003f;
    config->continuous_alpha = 0.2f;
    config->continuous_beta = 0.005f;
    config->continuous_gamma = 0.99f;
    
    // Set world model defaults (disabled by default)
    config->use_world_model = 0;
    config->latent_dim = 256;
    config->world_model_lr = 0.001f;
    config->world_model_update_freq = 100;
}

// Initialize enhanced model configuration
void init_enhanced_model_config(MuzeEnhancedModelConfig *model_config, 
                                MuzeEnhancedConfig *config) {
    if (!model_config || !config) return;
    
    // Initialize with zeros
    memset(model_config, 0, sizeof(MuzeEnhancedModelConfig));
    
    // Set observation dimension based on compressed obs
    model_config->obs_dim = (int)get_compressed_obs_size(config);
    model_config->latent_dim = config->latent_dim;
    model_config->action_count = get_total_discrete_actions(config);
    
    // Initialize NN config
    memset(&model_config->nn, 0, sizeof(MuNNConfig));
    
    // Set encoder defaults
    model_config->use_aabb = 1;
    model_config->use_silhouette = 1;
    model_config->use_voxel_grid = 1;
    model_config->use_local_grid = 1;
    model_config->use_agent_state = 1;
    
    // Set action encoding defaults
    model_config->pack_actions = 1;
    model_config->use_multi_discrete = 1;
}

// Initialize enhanced NN configuration
void init_enhanced_nn_config(MuzeEnhancedNNConfig *nn_config, 
                          MuzeEnhancedModelConfig *model_config) {
    if (!nn_config || !model_config) return;
    
    // Copy base config
    nn_config->base = model_config->nn;
    
    // Set encoder defaults
    nn_config->encoder_input_dim = model_config->obs_dim;
    nn_config->encoder_layers = 3;
    nn_config->encoder_hidden_dim = 256;
    nn_config->encoder_output_dim = model_config->latent_dim;
    
    // Set decoder defaults
    nn_config->decoder_input_dim = model_config->latent_dim;
    nn_config->decoder_layers = 3;
    nn_config->decoder_hidden_dim = 256;
    nn_config->decoder_output_dim = model_config->obs_dim;
    
    // Set actor defaults
    nn_config->obs_dim = model_config->obs_dim;
    nn_config->actor_layers = 3;
    nn_config->actor_hidden_dim = 256;
    nn_config->actor_output_dim = model_config->action_count;
    
    // Set critic defaults
    nn_config->critic_layers = 3;
    nn_config->critic_hidden_dim = 256;
    nn_config->critic_output_dim = 1;
    
    // Set world model defaults
    nn_config->world_model_layers = 3;
    nn_config->world_model_hidden_dim = 256;
    nn_config->world_model_output_dim = model_config->latent_dim;
}

// Validate enhanced configuration
int validate_enhanced_config(MuzeEnhancedConfig *config) {
    if (!config) return 0;
    
    // Validate action space configuration
    if (config->num_move_bins < 1 || config->num_move_bins > 100) {
        return 0;  // Invalid move bin count
    }
    
    if (config->num_forward_bins < 1 || config->num_forward_bins > 50) {
        return 0;  // Invalid forward bin count
    }
    
    if (config->num_turn_bins < 1 || config->num_turn_bins > 100) {
        return 0;  // Invalid turn bin count
    }
    
    // Validate observation configuration
    if (config->num_rays < 1 || config->num_rays > 512) {
        return 0;  // Invalid ray count
    }
    
    if (config->voxel_grid_size < 1 || config->voxel_grid_size > 64) {
        return 0;  // Invalid voxel grid size
    }
    
    if (config->local_grid_size < 1 || config->local_grid_size > 32) {
        return 0;  // Invalid local grid size
    }
    
    // Validate hybrid configuration
    if (config->use_continuous_actor && config->use_discrete_actions) {
        return 0;  // Cannot use both discrete and continuous actors
    }
    
    // Validate world model configuration
    if (config->use_world_model && config->latent_dim < 1) {
        return 0;  // Invalid latent dimension
    }
    
    return 1;  // Valid configuration
}

// Get total number of discrete actions
int get_total_discrete_actions(MuzeEnhancedConfig *config) {
    if (!config) return 0;
    
    if (config->pack_actions) {
        // Multi-discrete: move * forward * turn * attack * harvest
        return config->num_move_bins * config->num_forward_bins * 
               config->num_turn_bins * 2 * 2;
    } else {
        // Single discrete: just the move actions
        return config->num_move_bins;
    }
}

// Get compressed observation size
size_t get_compressed_obs_size(MuzeEnhancedConfig *config) {
    if (!config) return 0;
    
    size_t size = sizeof(CompressedObs);
    
    // Add space for ray data
    size += config->num_rays * 2 * sizeof(float);  // distances + angles
    
    // Add space for voxel grid
    size += config->voxel_grid_size * config->voxel_grid_size * 
             (config->voxel_grid_size / 2) * sizeof(uint8_t);
    
    // Add space for local grid
    size += config->local_grid_size * config->local_grid_size * sizeof(uint8_t);
    
    return size;
}
