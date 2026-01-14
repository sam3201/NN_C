#ifndef MUZE_ENHANCED_MODEL_H
#define MUZE_ENHANCED_MODEL_H

#include "muzero_model.h"
#include "discrete_actions.h"
#include "compressed_obs.h"
#include "muze_enhanced_config.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Enhanced MuZE model with discrete actions and compressed observations
typedef struct MuzeEnhancedModel {
    // Core MuZE model components
    MuModel *base_model;
    
    // Enhanced components
    CompressedObs *compressed_obs_buffer;
    
    // Configuration
    MuzeEnhancedConfig *config;
    MuzeEnhancedModelConfig *model_config;
    MuzeEnhancedNNConfig *nn_config;
    
    // Control flags
    int use_continuous_actor;
    
    // Encoder/Decoder networks (for compressed obs and latent space)
    NN_t *encoder;
    NN_t *decoder;
    
    // Actor/Critic networks (for continuous control)
    NN_t *actor;
    NN_t *critic;
    
    // World model (for large observations)
    NN_t *world_model;
    
    // Training state
    int training_step;
    float loss_value;
    
} MuzeEnhancedModel;

// Model creation and destruction
MuzeEnhancedModel *muze_enhanced_model_create(MuzeEnhancedConfig *config);
void muze_enhanced_model_destroy(MuzeEnhancedModel *model);

// Forward pass through enhanced model
void muze_enhanced_model_forward(MuzeEnhancedModel *model, CompressedObs *obs, 
                                int *action, float *value, float *policy);

// Training functions
void muze_enhanced_model_train_step(MuzeEnhancedModel *model, CompressedObs *obs, 
                               int action, float reward, int done);
void muze_enhanced_model_update_weights(MuzeEnhancedModel *model);

// Action encoding/decoding
int encode_action(float move, float forward, float turn, int attack, int harvest);
void decode_action(int action_id, float *move, float *forward, float *turn, int *attack, int *harvest);

// Observation compression/decompression
void compress_observation(float *raw_obs, int raw_size, CompressedObs *compressed);
void decompress_observation(CompressedObs *compressed, float *raw_obs, int raw_size);

// Model utilities
void muze_enhanced_model_print_summary(MuzeEnhancedModel *model);
size_t muze_enhanced_model_get_parameter_count(MuzeEnhancedModel *model);
int muze_enhanced_model_save(MuzeEnhancedModel *model, const char *filename);
MuzeEnhancedModel *muze_enhanced_model_load(const char *filename);

#ifdef __cplusplus
}
#endif

#endif // MUZE_ENHANCED_MODEL_H
