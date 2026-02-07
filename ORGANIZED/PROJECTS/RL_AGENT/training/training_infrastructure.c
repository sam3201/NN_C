#include "training_infrastructure.h"
#include <math.h>
#include <string.h>

// Create replay buffer
ReplayBuffer* replay_buffer_create(int capacity) {
    ReplayBuffer* buffer = malloc(sizeof(ReplayBuffer));
    if (!buffer) return NULL;
    
    buffer->experiences = malloc(sizeof(Experience) * capacity);
    if (!buffer->experiences) {
        free(buffer);
        return NULL;
    }
    
    buffer->capacity = capacity;
    buffer->size = 0;
    buffer->head = 0;
    buffer->tail = 0;
    
    pthread_mutex_init(&buffer->mutex, NULL);
    
    return buffer;
}

// Destroy replay buffer
void replay_buffer_destroy(ReplayBuffer* buffer) {
    if (!buffer) return;
    
    pthread_mutex_destroy(&buffer->mutex);
    if (buffer->experiences) free(buffer->experiences);
    free(buffer);
}

// Add experience to replay buffer
void replay_buffer_add(ReplayBuffer* buffer, Experience* exp) {
    if (!buffer || !exp) return;
    
    pthread_mutex_lock(&buffer->mutex);
    
    buffer->experiences[buffer->tail] = *exp;
    buffer->tail = (buffer->tail + 1) % buffer->capacity;
    
    if (buffer->size < buffer->capacity) {
        buffer->size++;
    } else {
        buffer->head = (buffer->head + 1) % buffer->capacity;
    }
    
    pthread_mutex_unlock(&buffer->mutex);
}

// Sample batch from replay buffer
int replay_buffer_sample(ReplayBuffer* buffer, Experience* batch, int batch_size) {
    if (!buffer || !batch || batch_size <= 0) return 0;
    
    pthread_mutex_lock(&buffer->mutex);
    
    int available = buffer->size;
    int actual_batch_size = (batch_size < available) ? batch_size : available;
    
    for (int i = 0; i < actual_batch_size; i++) {
        int idx = (buffer->head + rand() % available) % buffer->capacity;
        batch[i] = buffer->experiences[idx];
    }
    
    pthread_mutex_unlock(&buffer->mutex);
    
    return actual_batch_size;
}

// Clear replay buffer
void replay_buffer_clear(ReplayBuffer* buffer) {
    if (!buffer) return;
    
    pthread_mutex_lock(&buffer->mutex);
    buffer->size = 0;
    buffer->head = 0;
    buffer->tail = 0;
    pthread_mutex_unlock(&buffer->mutex);
}

// Create agent pool
AgentPool* agent_pool_create() {
    AgentPool* pool = malloc(sizeof(AgentPool));
    if (!pool) return NULL;
    
    pool->num_agents = 0;
    pool->current_main_agent = 0;
    
    pthread_mutex_init(&pool->mutex, NULL);
    
    return pool;
}

// Destroy agent pool
void agent_pool_destroy(AgentPool* pool) {
    if (!pool) return;
    
    pthread_mutex_lock(&pool->mutex);
    
    for (int i = 0; i < pool->num_agents; i++) {
        if (pool->agents[i]) {
            policy_value_network_destroy(pool->agents[i]);
        }
    }
    
    pthread_mutex_unlock(&pool->mutex);
    pthread_mutex_destroy(&pool->mutex);
    free(pool);
}

// Add agent to pool
void agent_pool_add_agent(AgentPool* pool, PolicyValueNetwork* agent) {
    if (!pool || !agent || pool->num_agents >= 10) return;
    
    pthread_mutex_lock(&pool->mutex);
    
    pool->agents[pool->num_agents] = agent;
    pool->num_agents++;
    
    pthread_mutex_unlock(&pool->mutex);
}

// Get opponent from pool
PolicyValueNetwork* agent_pool_get_opponent(AgentPool* pool) {
    if (!pool || pool->num_agents == 0) return NULL;
    
    pthread_mutex_lock(&pool->mutex);
    
    // Randomly select an opponent (could implement more sophisticated selection)
    int opponent_idx = rand() % pool->num_agents;
    PolicyValueNetwork* opponent = pool->agents[opponent_idx];
    
    pthread_mutex_unlock(&pool->mutex);
    
    return opponent;
}

// Create training statistics
TrainingStats* training_stats_create() {
    TrainingStats* stats = malloc(sizeof(TrainingStats));
    if (!stats) return NULL;
    
    stats->total_reward = 0.0f;
    stats->episode_length = 0.0f;
    stats->win_rate = 0.0f;
    stats->loss = 0.0f;
    stats->episodes_completed = 0;
    stats->total_steps = 0;
    stats->start_time = time(NULL);
    
    return stats;
}

// Update training statistics
void training_stats_update(TrainingStats* stats, float reward, int episode_length, int won) {
    if (!stats) return;
    
    stats->total_reward += reward;
    stats->episode_length = (stats->episode_length * stats->episodes_completed + episode_length) / (stats->episodes_completed + 1);
    stats->episodes_completed++;
    stats->win_rate = (stats->win_rate * (stats->episodes_completed - 1) + (won ? 1.0f : 0.0f)) / stats->episodes_completed;
}

// Print training statistics
void training_stats_print(TrainingStats* stats) {
    if (!stats) return;
    
    time_t current_time = time(NULL);
    double elapsed = difftime(current_time, stats->start_time);
    
    printf("\n=== Training Statistics ===\n");
    printf("Episodes Completed: %d\n", stats->episodes_completed);
    printf("Training Time: %.1f minutes\n", elapsed / 60.0);
    printf("Average Reward: %.2f\n", stats->total_reward / stats->episodes_completed);
    printf("Average Episode Length: %.1f\n", stats->episode_length);
    printf("Win Rate: %.2f%%\n", stats->win_rate * 100.0f);
    printf("Total Steps: %d\n", stats->total_steps);
    printf("Current Loss: %.6f\n", stats->loss);
    printf("========================\n");
}

// Destroy training statistics
void training_stats_destroy(TrainingStats* stats) {
    if (stats) {
        free(stats);
    }
}

// Get curriculum stage based on progress
CurriculumStage get_curriculum_stage(int episodes_completed) {
    if (episodes_completed < 100) return CURRICULUM_STAGE_0;
    if (episodes_completed < 300) return CURRICULUM_STAGE_1;
    if (episodes_completed < 500) return CURRICULUM_STAGE_2;
    return CURRICULUM_STAGE_3;
}

// Create environment based on curriculum stage
Environment create_environment(CurriculumStage stage) {
    Environment env;
    env.id = rand() % 1000;
    env.reward = 0.0f;
    env.done = 0;
    env.episode_step = 0;
    env.total_reward = 0.0f;
    
    // Initialize observation based on stage
    switch (stage) {
        case CURRICULUM_STAGE_0:
            // Environment only, no enemies
            memset(&env.current_obs, 0, sizeof(env.current_obs));
            break;
        case CURRICULUM_STAGE_1:
            // Passive enemies only
            memset(&env.current_obs, 0, sizeof(env.current_obs));
            // Add some passive mobs
            for (int i = 0; i < 5; i++) {
                int x = rand() % GRID_WIDTH;
                int y = rand() % GRID_HEIGHT;
                env.current_obs.grid[y][x][FEATURE_PASSIVE_MOB] = 1.0f;
            }
            break;
        case CURRICULUM_STAGE_2:
            // Weak enemies
            memset(&env.current_obs, 0, sizeof(env.current_obs));
            // Add weak hostile mobs
            for (int i = 0; i < 3; i++) {
                int x = rand() % GRID_WIDTH;
                int y = rand() % GRID_HEIGHT;
                env.current_obs.grid[y][x][FEATURE_HOSTILE_MOB] = 0.5f;  // Weak
            }
            break;
        case CURRICULUM_STAGE_3:
            // Full self-play
            memset(&env.current_obs, 0, sizeof(env.current_obs));
            // Add enemy agent
            int enemy_x = GRID_WIDTH - 5;
            int enemy_y = GRID_HEIGHT / 2;
            env.current_obs.grid[enemy_y][enemy_x][FEATURE_ENEMY_AGENT] = 1.0f;
            // Add resources
            for (int i = 0; i < 10; i++) {
                int x = rand() % GRID_WIDTH;
                int y = rand() % GRID_HEIGHT;
                env.current_obs.grid[y][x][FEATURE_RESOURCE] = 1.0f;
            }
            break;
    }
    
    // Add self agent
    env.current_obs.grid[GRID_HEIGHT/2][GRID_WIDTH/2][FEATURE_SELF_AGENT] = 1.0f;
    
    return env;
}

// Compute GAE (Generalized Advantage Estimation)
float compute_gae(Experience* trajectory, int length, float gamma, float lambda) {
    if (!trajectory || length <= 0) return 0.0f;
    
    float gae = 0.0f;
    float next_value = length > 1 ? trajectory[length - 1].value : 0.0f;
    
    for (int t = length - 1; t >= 0; t--) {
        float delta = trajectory[t].reward + gamma * next_value * (1 - trajectory[t].done) - trajectory[t].value;
        gae = delta + gamma * lambda * (1 - trajectory[t].done) * gae;
        trajectory[t].advantage = gae;
        next_value = trajectory[t].value;
    }
    
    return gae;
}

// Update policy using PPO
void update_policy(PolicyValueNetwork* policy, ReplayBuffer* replay_buffer, WorldModel* world_model) {
    if (!policy || !replay_buffer) return;
    
    // Sample batch from replay buffer
    Experience batch[BATCH_SIZE];
    int actual_batch_size = replay_buffer_sample(replay_buffer, batch, BATCH_SIZE);
    
    if (actual_batch_size == 0) return;
    
    // Prepare training data
    float states[actual_batch_size][LATENT_DIM];
    float actions[actual_batch_size][NUM_DISCRETE_MOVEMENTS + NUM_DISCRETE_ACTIONS + NUM_CONTINUOUS_ACTIONS];
    float old_values[actual_batch_size];
    float advantages[actual_batch_size];
    float returns[actual_batch_size];
    float log_probs[actual_batch_size];
    
    for (int i = 0; i < actual_batch_size; i++) {
        for (int j = 0; j < LATENT_DIM; j++) {
            states[i][j] = batch[i].state[j];
        }
        for (int j = 0; j < NUM_DISCRETE_MOVEMENTS + NUM_DISCRETE_ACTIONS + NUM_CONTINUOUS_ACTIONS; j++) {
            actions[i][j] = batch[i].action[j];
        }
        old_values[i] = batch[i].value;
        advantages[i] = batch[i].advantage;
        returns[i] = batch[i].value + batch[i].advantage;
        log_probs[i] = batch[i].log_prob;
    }
    
    // Normalize advantages
    float adv_mean = 0.0f;
    float adv_std = 0.0f;
    for (int i = 0; i < actual_batch_size; i++) {
        adv_mean += advantages[i];
    }
    adv_mean /= actual_batch_size;
    
    for (int i = 0; i < actual_batch_size; i++) {
        float diff = advantages[i] - adv_mean;
        adv_std += diff * diff;
    }
    adv_std = sqrtf(adv_std / actual_batch_size);
    
    for (int i = 0; i < actual_batch_size; i++) {
        advantages[i] = (advantages[i] - adv_mean) / (adv_std + 1e-8f);
    }
    
    // Perform PPO update (simplified - would normally use multiple epochs)
    float total_loss = 0.0f;
    for (int epoch = 0; epoch < PPO_EPOCHS; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int i = 0; i < actual_batch_size; i++) {
            // Forward pass through policy
            PolicyOutput output = policy_value_network_forward(policy, states[i]);
            
            // Calculate policy loss (simplified)
            float policy_loss = -(advantages[i] * log_probs[i]);
            
            // Calculate value loss
            float current_value = policy_value_network_get_value(policy);
            float value_loss = (current_value - returns[i]) * (current_value - returns[i]);
            
            // Calculate entropy bonus
            float entropy = 0.0f; // Would calculate actual entropy from action distributions
            
            // Total loss
            float loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy;
            epoch_loss += loss;
        }
        
        total_loss += epoch_loss / actual_batch_size;
    }
    
    // Update training statistics
    // Note: In a real implementation, we would actually update the network weights here
    // This is a simplified version that just tracks the loss
}

// Training worker thread
void* training_worker_thread(void* arg) {
    TrainingThreadData* data = (TrainingThreadData*)arg;
    
    printf("Training worker %d started\n", data->thread_id);
    
    while (*data->training_active) {
        training_loop(data);
        usleep(100000); // 100ms delay
    }
    
    printf("Training worker %d stopped\n", data->thread_id);
    return NULL;
}

// Main training loop
void training_loop(TrainingThreadData* data) {
    // Get curriculum stage
    CurriculumStage stage = get_curriculum_stage(data->stats->episodes_completed);
    
    // Create environment
    Environment env = create_environment(stage);
    
    // Initialize episode
    float episode_reward = 0.0f;
    int episode_steps = 0;
    
    while (episode_steps < MAX_EPISODE_LENGTH && *data->training_active) {
        // Encode observation
        float latent[LATENT_DIM];
        observation_encoder_encode(data->encoder, &env.current_obs, latent);
        
        // Get policy action
        PolicyOutput policy_output = policy_value_network_forward(data->policy, latent);
        
        // Execute action (simplified environment step)
        env.reward = 0.0f; // Would be determined by actual game logic
        env.done = (episode_steps >= MAX_EPISODE_LENGTH - 1);
        
        // Get next observation (simplified)
        if (!env.done) {
            // Would get actual next observation from environment
            memset(&env.current_obs, 0, sizeof(env.current_obs));
            env.current_obs.grid[GRID_HEIGHT/2][GRID_WIDTH/2][FEATURE_SELF_AGENT] = 1.0f;
        }
        
        episode_reward += env.reward;
        episode_steps++;
        data->stats->total_steps++;
        
        // Store experience
        Experience exp;
        memcpy(exp.state, latent, sizeof(float) * LATENT_DIM);
        exp.action[0] = (float)policy_output.sampled_movement;
        exp.action[1] = (float)policy_output.sampled_action;
        exp.action[2] = policy_output.sampled_aim[0];
        exp.action[3] = policy_output.sampled_aim[1];
        exp.action[4] = policy_output.sampled_charge;
        exp.reward = env.reward;
        exp.value = policy_value_network_get_value(data->policy);
        exp.done = env.done;
        exp.log_prob = policy_output.movement_log_prob + policy_output.action_log_prob + 
                     policy_output.aim_log_prob + policy_output.charge_log_prob;
        
        replay_buffer_add(data->replay_buffer, &exp);
        
        // Update environment
        env.episode_step = episode_steps;
        env.total_reward = episode_reward;
        
        // Check if episode is done
        if (env.done) {
            training_stats_update(data->stats, episode_reward, episode_steps, 1); // Simplified win condition
            data->episodes_completed++;
            
            // Update policy periodically
            if (data->episodes_completed % 10 == 0) {
                update_policy(data->policy, data->replay_buffer, data->world_model);
            }
            
            // Reset for next episode
            stage = get_curriculum_stage(data->stats->episodes_completed);
            env = create_environment(stage);
            episode_reward = 0.0f;
            episode_steps = 0;
        }
    }
}
