#ifndef RL_AGENT_H
#define RL_AGENT_H

#include "encoder/observation_encoder.h"
#include "policy/policy_value_network.h"
#include "world_model/world_model.h"
#include "training/training_infrastructure.h"

// Main RL Agent structure
typedef struct {
    // Core components
    ObservationEncoder* encoder;
    PolicyValueNetwork* policy;
    WorldModel* world_model;
    
    // Training infrastructure
    ReplayBuffer* replay_buffer;
    AgentPool* agent_pool;
    TrainingStats* training_stats;
    
    // Training threads
    pthread_t* training_threads;
    int num_training_threads;
    volatile int training_active;
    
    // Configuration
    int use_world_model;
    int use_self_play;
    int curriculum_learning;
    
    // Agent state
    float current_latent[LATENT_DIM];
    PolicyOutput last_action;
    int episode_step;
    float episode_reward;
    
    // Thread synchronization
    pthread_mutex_t state_mutex;
} RLAgent;

// Agent configuration
typedef struct {
    int use_world_model;
    int use_self_play;
    int curriculum_learning;
    int num_training_threads;
    float learning_rate;
    int batch_size;
    int replay_buffer_size;
    int max_episode_length;
} AgentConfig;

// Function declarations
RLAgent* rl_agent_create(AgentConfig* config);
void rl_agent_destroy(RLAgent* agent);
void rl_agent_reset(RLAgent* agent);
PolicyOutput rl_agent_act(RLAgent* agent, GridObservation* obs);
void rl_agent_update(RLAgent* agent, float reward, GridObservation* next_obs, int done);
void rl_agent_start_training(RLAgent* agent);
void rl_agent_stop_training(RLAgent* agent);
void rl_agent_print_stats(RLAgent* agent);
int rl_agent_is_training(RLAgent* agent);

// Default configuration
AgentConfig get_default_config();

#endif // RL_AGENT_H
