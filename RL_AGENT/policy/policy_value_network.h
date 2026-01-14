#ifndef POLICY_VALUE_NETWORK_H
#define POLICY_VALUE_NETWORK_H

#include "../encoder/observation_encoder.h"

// Action space definitions
#define NUM_DISCRETE_MOVEMENTS 5  // idle, forward, backward, left, right, jump
#define NUM_DISCRETE_ACTIONS 3     // idle, attack, harvest
#define NUM_CONTINUOUS_ACTIONS 3  // aim_x, aim_y, charge_intensity

// Recurrent core type
typedef enum {
    RECURRENT_LSTM,
    RECURRENT_TRANSFORMER
} RecurrentType;

// Policy output structure
typedef struct {
    // Discrete action logits
    float movement_logits[NUM_DISCRETE_MOVEMENTS];
    float action_logits[NUM_DISCRETE_ACTIONS];
    
    // Continuous action parameters (mean and std for Gaussian)
    float aim_mean[2];      // aim_x, aim_y
    float aim_std[2];       // std for aim_x, aim_y
    float charge_mean;      // charge intensity
    float charge_std;       // std for charge intensity
    
    // Sampled actions
    int sampled_movement;
    int sampled_action;
    float sampled_aim[2];
    float sampled_charge;
    
    // Log probabilities for training
    float movement_log_prob;
    float action_log_prob;
    float aim_log_prob;
    float charge_log_prob;
} PolicyOutput;

// Policy and value network state
typedef struct {
    // Shared core network
    NN_t* shared_core;
    
    // Recurrent component (LSTM or Transformer)
    RecurrentType recurrent_type;
    void* recurrent_core;  // Will be cast to appropriate type
    
    // Policy heads
    NN_t* movement_head;
    NN_t* action_head;
    NN_t* aim_mean_head;
    NN_t* aim_std_head;
    NN_t* charge_mean_head;
    NN_t* charge_std_head;
    
    // Value head
    NN_t* value_head;
    
    // Hidden state for recurrent component
    float* hidden_state;
    int hidden_state_size;
    
    // Current latent input
    float current_latent[LATENT_DIM];
    
    // Training mode
    int training_mode;
} PolicyValueNetwork;

// Function declarations
PolicyValueNetwork* policy_value_network_create(RecurrentType recurrent_type);
void policy_value_network_destroy(PolicyValueNetwork* network);
PolicyOutput policy_value_network_forward(PolicyValueNetwork* network, float* latent_vector);
void policy_value_network_reset(PolicyValueNetwork* network);
float policy_value_network_get_value(PolicyValueNetwork* network);
void policy_value_network_set_training_mode(PolicyValueNetwork* network, int training_mode);

// Action sampling functions
int sample_categorical(float* logits, int num_categories);
void sample_gaussian(float mean, float std, float* sample);
float categorical_log_prob(float* logits, int category, int num_categories);
float gaussian_log_prob(float sample, float mean, float std);

#endif // POLICY_VALUE_NETWORK_H
