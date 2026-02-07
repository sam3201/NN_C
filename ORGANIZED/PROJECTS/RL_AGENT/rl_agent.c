#include "rl_agent.h"
#include <string.h>

// Get default configuration
AgentConfig get_default_config() {
  AgentConfig config;
  config.use_world_model = 1;
  config.use_self_play = 1;
  config.curriculum_learning = 1;
  config.num_training_threads = 4;
  config.learning_rate = 3e-4f;
  config.batch_size = 64;
  config.replay_buffer_size = 100000;
  config.max_episode_length = 1000;
  return config;
}

// Create RL agent
RLAgent *rl_agent_create(AgentConfig *config) {
  RLAgent *agent = malloc(sizeof(RLAgent));
  if (!agent)
    return NULL;

  // Store configuration
  agent->use_world_model = config->use_world_model;
  agent->use_self_play = config->use_self_play;
  agent->curriculum_learning = config->curriculum_learning;
  agent->num_training_threads = config->num_training_threads;
  agent->training_active = 0;
  agent->episode_step = 0;
  agent->episode_reward = 0.0f;

  // Initialize mutex
  pthread_mutex_init(&agent->state_mutex, NULL);

  // Create observation encoder
  agent->encoder = observation_encoder_create();
  if (!agent->encoder) {
    free(agent);
    return NULL;
  }

  // Create policy and value network
  agent->policy = policy_value_network_create(RECURRENT_LSTM);
  if (!agent->policy) {
    observation_encoder_destroy(agent->encoder);
    free(agent);
    return NULL;
  }

  // Create world model if enabled
  if (config->use_world_model) {
    agent->world_model = world_model_create(1); // Use transformer
    if (!agent->world_model) {
      policy_value_network_destroy(agent->policy);
      observation_encoder_destroy(agent->encoder);
      free(agent);
      return NULL;
    }
  } else {
    agent->world_model = NULL;
  }

  // Create replay buffer
  agent->replay_buffer = replay_buffer_create(config->replay_buffer_size);
  if (!agent->replay_buffer) {
    if (agent->world_model)
      world_model_destroy(agent->world_model);
    policy_value_network_destroy(agent->policy);
    observation_encoder_destroy(agent->encoder);
    free(agent);
    return NULL;
  }

  // Create agent pool for self-play
  if (config->use_self_play) {
    agent->agent_pool = agent_pool_create();
    if (!agent->agent_pool) {
      replay_buffer_destroy(agent->replay_buffer);
      if (agent->world_model)
        world_model_destroy(agent->world_model);
      policy_value_network_destroy(agent->policy);
      observation_encoder_destroy(agent->encoder);
      free(agent);
      return NULL;
    }

    // Add initial agent to pool
    agent_pool_add_agent(agent->agent_pool, agent->policy);
  } else {
    agent->agent_pool = NULL;
  }

  // Create training statistics
  agent->training_stats = training_stats_create();
  if (!agent->training_stats) {
    if (agent->agent_pool)
      agent_pool_destroy(agent->agent_pool);
    replay_buffer_destroy(agent->replay_buffer);
    if (agent->world_model)
      world_model_destroy(agent->world_model);
    policy_value_network_destroy(agent->policy);
    observation_encoder_destroy(agent->encoder);
    free(agent);
    return NULL;
  }

  // Allocate training threads
  agent->training_threads =
      malloc(sizeof(pthread_t) * config->num_training_threads);
  if (!agent->training_threads) {
    training_stats_destroy(agent->training_stats);
    if (agent->agent_pool)
      agent_pool_destroy(agent->agent_pool);
    replay_buffer_destroy(agent->replay_buffer);
    if (agent->world_model)
      world_model_destroy(agent->world_model);
    policy_value_network_destroy(agent->policy);
    observation_encoder_destroy(agent->encoder);
    free(agent);
    return NULL;
  }

  // Initialize agent state
  memset(agent->current_latent, 0, sizeof(float) * LATENT_DIM);
  memset(&agent->last_action, 0, sizeof(PolicyOutput));

  printf("RL Agent created successfully!\n");
  printf("Configuration:\n");
  printf("  World Model: %s\n",
         config->use_world_model ? "Enabled" : "Disabled");
  printf("  Self-Play: %s\n", config->use_self_play ? "Enabled" : "Disabled");
  printf("  Curriculum Learning: %s\n",
         config->curriculum_learning ? "Enabled" : "Disabled");
  printf("  Training Threads: %d\n", config->num_training_threads);
  printf("  Replay Buffer Size: %d\n", config->replay_buffer_size);
  printf("  Max Episode Length: %d\n", config->max_episode_length);

  return agent;
}

// Destroy RL agent
void rl_agent_destroy(RLAgent *agent) {
  if (!agent)
    return;

  // Stop training if active
  rl_agent_stop_training(agent);

  // Wait for threads to finish
  for (int i = 0; i < agent->num_training_threads; i++) {
    pthread_join(agent->training_threads[i], NULL);
  }

  // Clean up components
  if (agent->encoder)
    observation_encoder_destroy(agent->encoder);
  if (agent->policy)
    policy_value_network_destroy(agent->policy);
  if (agent->world_model)
    world_model_destroy(agent->world_model);
  if (agent->replay_buffer)
    replay_buffer_destroy(agent->replay_buffer);
  if (agent->agent_pool)
    agent_pool_destroy(agent->agent_pool);
  if (agent->training_stats)
    training_stats_destroy(agent->training_stats);
  if (agent->training_threads)
    free(agent->training_threads);

  pthread_mutex_destroy(&agent->state_mutex);

  free(agent);
}

// Reset agent state
void rl_agent_reset(RLAgent *agent) {
  if (!agent)
    return;

  pthread_mutex_lock(&agent->state_mutex);

  observation_encoder_reset(agent->encoder);
  policy_value_network_reset(agent->policy);
  if (agent->world_model) {
    world_model_reset(agent->world_model);
  }

  memset(agent->current_latent, 0, sizeof(float) * LATENT_DIM);
  memset(&agent->last_action, 0, sizeof(PolicyOutput));
  agent->episode_step = 0;
  agent->episode_reward = 0.0f;

  pthread_mutex_unlock(&agent->state_mutex);
}

// Agent action selection
PolicyOutput rl_agent_act(RLAgent *agent, GridObservation *obs) {
  PolicyOutput output;

  if (!agent || !obs) {
    memset(&output, 0, sizeof(PolicyOutput));
    return output;
  }

  pthread_mutex_lock(&agent->state_mutex);

  // Encode observation
  observation_encoder_encode(agent->encoder, obs, agent->current_latent);

  // Get policy action
  output = policy_value_network_forward(agent->policy, agent->current_latent);

  // Store last action
  agent->last_action = output;
  agent->episode_step++;

  pthread_mutex_unlock(&agent->state_mutex);

  return output;
}

// Update agent with environment feedback
void rl_agent_update(RLAgent *agent, float reward, GridObservation *next_obs,
                     int done) {
  if (!agent)
    return;

  pthread_mutex_lock(&agent->state_mutex);

  agent->episode_reward += reward;

  // Encode next observation
  float next_latent[LATENT_DIM];
  if (next_obs) {
    observation_encoder_encode(agent->encoder, next_obs, next_latent);
  }

  // Create experience for replay buffer
  Experience exp;
  memcpy(exp.state, agent->current_latent, sizeof(float) * LATENT_DIM);
  exp.action[0] = (float)agent->last_action.sampled_movement;
  exp.action[1] = (float)agent->last_action.sampled_action;
  exp.action[2] = agent->last_action.sampled_aim[0];
  exp.action[3] = agent->last_action.sampled_aim[1];
  exp.action[4] = agent->last_action.sampled_charge;
  exp.reward = reward;
  exp.value = policy_value_network_get_value(agent->policy);
  exp.done = done;
  exp.log_prob = agent->last_action.movement_log_prob +
                 agent->last_action.action_log_prob +
                 agent->last_action.aim_log_prob +
                 agent->last_action.charge_log_prob;

  // Add to replay buffer
  replay_buffer_add(agent->replay_buffer, &exp);

  // Update current state
  if (next_obs) {
    memcpy(agent->current_latent, next_latent, sizeof(float) * LATENT_DIM);
  }

  // Reset if episode is done
  if (done) {
    training_stats_update(agent->training_stats, agent->episode_reward,
                          agent->episode_step, 0);
    agent->episode_step = 0;
    agent->episode_reward = 0.0f;
  }

  pthread_mutex_unlock(&agent->state_mutex);
}

// Start training
void rl_agent_start_training(RLAgent *agent) {
  if (!agent || agent->training_active)
    return;

  printf("Starting RL agent training...\n");

  agent->training_active = 1;

  // Create training thread data
  TrainingThreadData *thread_data =
      malloc(sizeof(TrainingThreadData) * agent->num_training_threads);
  for (int i = 0; i < agent->num_training_threads; i++) {
    thread_data[i].thread_id = i;
    thread_data[i].encoder = agent->encoder;
    thread_data[i].policy = agent->policy;
    thread_data[i].world_model = agent->world_model;
    thread_data[i].replay_buffer = agent->replay_buffer;
    thread_data[i].agent_pool = agent->agent_pool;
    thread_data[i].stats = agent->training_stats;
    thread_data[i].training_active = &agent->training_active;
    thread_data[i].episodes_completed = 0;
  }

  // Start training threads
  for (int i = 0; i < agent->num_training_threads; i++) {
    pthread_create(&agent->training_threads[i], NULL, training_worker_thread,
                   &thread_data[i]);
  }

  printf("Training started with %d threads\n", agent->num_training_threads);
}

// Stop training
void rl_agent_stop_training(RLAgent *agent) {
  if (!agent || !agent->training_active)
    return;

  printf("Stopping RL agent training...\n");

  agent->training_active = 0;

  // Threads will clean up automatically
}

// Print training statistics
void rl_agent_print_stats(RLAgent *agent) {
  if (!agent)
    return;

  pthread_mutex_lock(&agent->state_mutex);
  training_stats_print(agent->training_stats);
  pthread_mutex_unlock(&agent->state_mutex);
}

// Check if agent is training
int rl_agent_is_training(RLAgent *agent) {
  return agent ? agent->training_active : 0;
}
