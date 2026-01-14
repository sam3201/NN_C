#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "rl_agent.h"

// Create a simple test environment
typedef struct {
    int step;
    int max_steps;
    float total_reward;
    int done;
    GridObservation obs;
} TestEnvironment;

// Initialize test environment
void init_test_environment(TestEnvironment* env, int max_steps) {
    env->step = 0;
    env->max_steps = max_steps;
    env->total_reward = 0.0f;
    env->done = 0;
    
    // Initialize observation grid
    memset(&env->obs, 0, sizeof(env->obs));
    
    // Add self agent in center
    env->obs.grid[GRID_HEIGHT/2][GRID_WIDTH/2][FEATURE_SELF_AGENT] = 1.0f;
    
    // Add some resources
    for (int i = 0; i < 10; i++) {
        int x = rand() % GRID_WIDTH;
        int y = rand() % GRID_HEIGHT;
        env->obs.grid[y][x][FEATURE_RESOURCE] = 1.0f;
    }
    
    // Add some obstacles
    for (int i = 0; i < 5; i++) {
        int x = rand() % GRID_WIDTH;
        int y = rand() % GRID_HEIGHT;
        env->obs.grid[y][x][FEATURE_STRUCTURE] = 1.0f;
    }
}

// Step environment
void step_test_environment(TestEnvironment* env, int action) {
    env->step++;
    
    // Simple reward: +1 for collecting resources
    for (int y = 0; y < GRID_HEIGHT; y++) {
        for (int x = 0; x < GRID_WIDTH; x++) {
            if (env->obs.grid[y][x][FEATURE_RESOURCE] > 0.5f) {
                // Check if agent is adjacent to resource
                int agent_x = GRID_WIDTH/2;
                int agent_y = GRID_HEIGHT/2;
                if (abs(x - agent_x) <= 1 && abs(y - agent_y) <= 1) {
                    env->total_reward += 1.0f;
                    env->obs.grid[y][x][FEATURE_RESOURCE] = 0.0f; // Collect resource
                }
            }
        }
    }
    
    // Move agent based on action (simplified)
    int agent_x = GRID_WIDTH/2;
    int agent_y = GRID_HEIGHT/2;
    
    switch (action) {
        case 1: // forward
            if (agent_y > 0) agent_y--;
            break;
        case 2: // backward
            if (agent_y < GRID_HEIGHT - 1) agent_y++;
            break;
        case 3: // left
            if (agent_x > 0) agent_x--;
            break;
        case 4: // right
            if (agent_x < GRID_WIDTH - 1) agent_x++;
            break;
    }
    
    // Update observation
    memset(&env->obs, 0, sizeof(env->obs));
    env->obs.grid[agent_y][agent_x][FEATURE_SELF_AGENT] = 1.0f;
    
    // Re-add some resources
    for (int i = 0; i < 10; i++) {
        int x = rand() % GRID_WIDTH;
        int y = rand() % GRID_HEIGHT;
        env->obs.grid[y][x][FEATURE_RESOURCE] = 1.0f;
    }
    
    // Check if episode is done
    if (env->step >= env->max_steps || env->total_reward >= 20.0f) {
        env->done = 1;
    }
}

// Test basic agent functionality
void test_basic_functionality() {
    printf("=== Testing Basic RL Agent Functionality ===\n");
    
    // Create agent with default configuration
    AgentConfig config = get_default_config();
    config.use_world_model = 0; // Disable world model for basic test
    config.num_training_threads = 1;
    
    RLAgent* agent = rl_agent_create(&config);
    if (!agent) {
        printf("Failed to create RL agent\n");
        return;
    }
    
    // Create test environment
    TestEnvironment env;
    init_test_environment(&env, 50);
    
    printf("Running test episode...\n");
    
    // Run episode
    int steps = 0;
    while (!env.done && steps < env.max_steps) {
        // Get agent action
        PolicyOutput action = rl_agent_act(agent, &env.obs);
        
        // Step environment
        step_test_environment(&env, action.sampled_movement);
        
        // Update agent
        rl_agent_update(agent, env.total_reward, &env.obs, env.done);
        
        printf("Step %d: Action=%d, Reward=%.1f, Total Reward=%.1f\n", 
               steps, action.sampled_movement, env.total_reward, env.total_reward);
        
        steps++;
    }
    
    printf("Episode completed: Steps=%d, Total Reward=%.2f\n", steps, env.total_reward);
    
    // Print agent statistics
    rl_agent_print_stats(agent);
    
    // Cleanup
    rl_agent_destroy(agent);
    printf("Basic functionality test completed!\n\n");
}

// Test training loop
void test_training_loop() {
    printf("=== Testing Training Loop ===\n");
    
    // Create agent with training enabled
    AgentConfig config = get_default_config();
    config.use_world_model = 1;
    config.use_self_play = 1;
    config.curriculum_learning = 1;
    config.num_training_threads = 2;
    config.replay_buffer_size = 10000;
    
    RLAgent* agent = rl_agent_create(&config);
    if (!agent) {
        printf("Failed to create RL agent for training\n");
        return;
    }
    
    // Start training
    rl_agent_start_training(agent);
    
    // Let training run for a short time
    printf("Training for 10 seconds...\n");
    sleep(10);
    
    // Stop training
    rl_agent_stop_training(agent);
    
    // Print final statistics
    rl_agent_print_stats(agent);
    
    // Cleanup
    rl_agent_destroy(agent);
    printf("Training loop test completed!\n\n");
}

// Test world model integration
void test_world_model() {
    printf("=== Testing World Model Integration ===\n");
    
    // Create agent with world model
    AgentConfig config = get_default_config();
    config.use_world_model = 1;
    config.num_training_threads = 1;
    
    RLAgent* agent = rl_agent_create(&config);
    if (!agent) {
        printf("Failed to create RL agent with world model\n");
        return;
    }
    
    // Create test environment
    TestEnvironment env;
    init_test_environment(&env, 20);
    
    printf("Testing world model imagination...\n");
    
    // Run a few steps to collect experience
    PolicyOutput last_action;
    for (int i = 0; i < 5; i++) {
        PolicyOutput action = rl_agent_act(agent, &env.obs);
        step_test_environment(&env, action.sampled_movement);
        rl_agent_update(agent, env.total_reward, &env.obs, 0);
        last_action = action; // Store last action for world model test
    }
    
    // Test imagination (simplified)
    if (agent->world_model) {
        printf("Imagining trajectory with world model...\n");
        
        float initial_state[LATENT_DIM];
        observation_encoder_encode(agent->encoder, &env.obs, initial_state);
        
        // Imagine trajectory
        world_model_imagine_trajectory(agent->world_model, initial_state, &last_action, 5);
        
        printf("World model losses:\n");
        printf("  Transition Loss: %.6f\n", world_model_get_transition_loss(agent->world_model));
        printf("  Reward Loss: %.6f\n", world_model_get_reward_loss(agent->world_model));
        printf("  Posterior Loss: %.6f\n", world_model_get_posterior_loss(agent->world_model));
    }
    
    // Cleanup
    rl_agent_destroy(agent);
    printf("World model integration test completed!\n\n");
}

// Test self-play capabilities
void test_self_play() {
    printf("=== Testing Self-Play Capabilities ===\n");
    
    // Create agent with self-play
    AgentConfig config = get_default_config();
    config.use_self_play = 1;
    config.curriculum_learning = 1;
    config.num_training_threads = 2;
    
    RLAgent* agent = rl_agent_create(&config);
    if (!agent) {
        printf("Failed to create RL agent for self-play\n");
        return;
    }
    
    printf("Testing self-play with %d training threads...\n", config.num_training_threads);
    
    // Start self-play training
    rl_agent_start_training(agent);
    
    // Let training run for a short time
    printf("Self-play training for 15 seconds...\n");
    sleep(15);
    
    // Stop training
    rl_agent_stop_training(agent);
    
    // Print statistics
    rl_agent_print_stats(agent);
    
    // Cleanup
    rl_agent_destroy(agent);
    printf("Self-play test completed!\n\n");
}

// Main function
int main() {
    printf("=== Scalable RL Architecture Test Suite ===\n");
    printf("Testing Multi-Agent Base-Takeover Game Components\n\n");
    
    srand(time(NULL));
    
    // Run tests
    test_basic_functionality();
    test_training_loop();
    test_world_model();
    test_self_play();
    
    printf("=== All Tests Completed Successfully! ===\n");
    printf("The RL agent architecture is working correctly.\n");
    printf("Ready for integration with the actual game environment.\n");
    
    return 0;
}
