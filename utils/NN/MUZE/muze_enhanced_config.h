#ifndef MUZE_ENHANCED_CONFIG_H
#define MUZE_ENHANCED_CONFIG_H

#include "muzero_model.h"
#include "discrete_actions.h"
#include "compressed_obs.h"
#include "muze_config.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Enhanced configuration for discrete actions and compressed observations
typedef struct {
    // Enhanced action space configuration
    int use_discrete_actions;        // Use discretized continuous actions
    int num_move_bins;              // Number of bins for movement (default: 31)
    int num_forward_bins;           // Number of bins for forward (default: 11)
    int num_turn_bins;              // Number of bins for turning (default: 31)
    int pack_actions;               // Pack multiple actions into single ID
    
    // Compressed observation configuration
    int use_compressed_obs;          // Use compressed observations
    int num_rays;                  // Number of ray samples (default: 128)
    int voxel_grid_size;             // Coarse voxel grid size (default: 16)
    int local_grid_size;             // Local grid size (default: 8)
    int max_token_value;             // Maximum token value for local grid (default: 7)
    
    // Hybrid configuration (for future SAC integration)
    int use_continuous_actor;       // Use continuous actor-critic instead of MuZE
    float continuous_lr;             // Learning rate for continuous actor
    float continuous_alpha;           // Alpha for soft actor updates
    float continuous_beta;            // Beta for soft actor updates
    float continuous_gamma;           // Gamma for advantage estimation
    
    // World model configuration (for future Dreamer integration)
    int use_world_model;             // Use latent world model
    int latent_dim;                  // Latent space dimension
    float world_model_lr;            // Learning rate for world model
    int world_model_update_freq;      // Frequency of world model updates
    
} MuzeEnhancedConfig;

// Enhanced model configuration with compressed observations
typedef struct {
    int obs_dim;                    // Compressed observation dimension
    int latent_dim;                 // Latent dimension (if using world model)
    int action_count;               // Total discrete action count
    MuNNConfig nn;                  // NN configuration
    
    // Observation encoder configuration
    int use_aabb;                   // Extract AABB from vertices
    int use_silhouette;             // Extract silhouette from depth
    int use_voxel_grid;             // Extract coarse voxel grid
    int use_local_grid;             // Tokenize local observation
    int use_agent_state;            // Include agent health/energy/etc
    
    // Action encoder configuration
    int pack_actions;               // Pack multiple actions into single ID
    int use_multi_discrete;         // Use multi-discrete action space
    
} MuzeEnhancedModelConfig;

// Enhanced NN configuration
typedef struct {
    // Standard NN config
    MuNNConfig base;
    
    // Encoder network (for compressed obs)
    int encoder_input_dim;
    int encoder_layers;
    int encoder_hidden_dim;
    int encoder_output_dim;
    
    // Decoder network (for latent space)
    int decoder_input_dim;
    int decoder_layers;
    int decoder_hidden_dim;
    int decoder_output_dim;
    
    // Actor network (policy)
    int obs_dim;
    int actor_layers;
    int actor_hidden_dim;
    int actor_output_dim;
    
    // Critic network (value)
    int critic_layers;
    int critic_hidden_dim;
    int critic_output_dim;
    
    // World model (if enabled)
    int world_model_layers;
    int world_model_hidden_dim;
    int world_model_output_dim;
    
} MuzeEnhancedNNConfig;

// Initialize enhanced configuration
void init_enhanced_config(MuzeEnhancedConfig *config);
void init_enhanced_model_config(MuzeEnhancedModelConfig *model_config, 
                                MuzeEnhancedConfig *config);
void init_enhanced_nn_config(MuzeEnhancedNNConfig *nn_config, 
                          MuzeEnhancedModelConfig *model_config);

// Configuration validation
int validate_enhanced_config(MuzeEnhancedConfig *config);

// Utility functions
int get_total_discrete_actions(MuzeEnhancedConfig *config);
size_t get_compressed_obs_size(MuzeEnhancedConfig *config);

#ifdef __cplusplus
}
#endif

#endif // MUZE_ENHANCED_CONFIG_H
