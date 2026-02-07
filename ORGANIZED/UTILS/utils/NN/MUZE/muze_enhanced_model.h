#ifndef MUZE_ENHANCED_MODEL_H
#define MUZE_ENHANCED_MODEL_H

#include "muzero_model.h"
#include "discrete_actions.h"
#include "compressed_obs.h"
#include "muze_enhanced_config.h"
#include "../NN/NN.h"
#include <stddef.h>

// Dominant Compression Principle Integration
// Based on AM's variational framework: arg max E[τ] - βH - λC + ηI
// Standalone implementation without PyTorch dependency
#define DOMINANT_COMPRESSION_BETA 1.0      // β - Uncertainty weight
#define DOMINANT_COMPRESSION_LAMBDA 0.1    // λ - Compute cost weight
#define DOMINANT_COMPRESSION_ETA 0.01      // η - Useful memory weight
#define DOMINANT_COMPRESSION_KAPPA 0.1     // κ - Required return on compute

#ifdef __cplusplus
extern "C" {
#endif

// Enhanced MuZE model with Dominant Compression
typedef struct {
    // Core MuZE model components
    MuModel *base_model;

    // Enhanced components
    CompressedObs *compressed_obs_buffer;

    // Dominant Compression components (standalone implementation)
    double *policy;                    // π - Policy (action selection)
    double *memory;                    // M - Memory/context system
    double *world_model_dc;            // θ - Standalone world model (predictive dynamics)
    double resource_alloc;             // ρ - Resource allocator
    double uncertainty;                // H - Predictive uncertainty
    double compute_cost;               // C - Compute/capacity cost
    double mutual_info;                // I - Useful memory
    double objective;                  // J - Main objective
    double capacity;                   // Current capacity
    int learning_plateau;              // Eval cycles with plateau

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

// Standalone Dominant Compression functions (no PyTorch)
double compute_dominant_objective_standalone(MuzeEnhancedModel *model);
double predict_uncertainty_standalone(double *states, double *next_states, int dim);
double compute_mutual_info_standalone(double *memory, double *states, int mem_dim, int state_dim);
int should_grow_capacity_standalone(MuzeEnhancedModel *model, double delta_J, double delta_C);
void update_capacity_standalone(MuzeEnhancedModel *model, double performance_gain, double compute_increase);
void distill_policy_standalone(MuzeEnhancedModel *model);

// Initialize with Dominant Compression parameters
void init_dominant_compression_standalone(MuzeEnhancedModel *model);

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
