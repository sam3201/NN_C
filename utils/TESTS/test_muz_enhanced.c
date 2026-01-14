#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

// Include enhanced MUZE headers
#include "../NN/MUZE/muze_enhanced_config.h"
#include "../NN/MUZE/muze_enhanced_model.h"
#include "../NN/MUZE/discrete_actions.h"
#include "../NN/MUZE/compressed_obs.h"

// Test helper functions
static int test_passed = 0;
static int test_failed = 0;

#define TEST_ASSERT(condition, message) \
    do { \
        if (condition) { \
            printf("✓ PASS: %s\n", message); \
            test_passed++; \
        } else { \
            printf("✗ FAIL: %s\n", message); \
            test_failed++; \
        } \
    } while(0)

// Test discrete action encoding/decoding
void test_discrete_actions() {
    printf("=== Testing Discrete Actions ===\n");
    
    // Test action packing
    float move = 0.5f, forward = 0.8f, turn = -0.3f;
    int attack = 1, harvest = 0;
    
    int packed_action = encode_action(move, forward, turn, attack, harvest);
    TEST_ASSERT(packed_action >= 0 && packed_action < TOTAL_DISCRETE_ACTIONS, "Action packing works");
    
    // Test action unpacking
    float unpacked_move, unpacked_forward, unpacked_turn;
    int unpacked_attack, unpacked_harvest;
    
    decode_action(packed_action, &unpacked_move, &unpacked_forward, &unpacked_turn, &unpacked_attack, &unpacked_harvest);
    
    TEST_ASSERT(fabsf(unpacked_move - move) < 0.01f, "Move unpacking works");
    TEST_ASSERT(fabsf(unpacked_forward - forward) < 0.01f, "Forward unpacking works");
    TEST_ASSERT(fabsf(unpacked_turn - turn) < 0.01f, "Turn unpacking works");
    TEST_ASSERT(unpacked_attack == attack, "Attack unpacking works");
    TEST_ASSERT(unpacked_harvest == harvest, "Harvest unpacking works");
    
    printf("✓ Discrete actions test passed\n");
}

// Test compressed observations
void test_compressed_observations() {
    printf("=== Testing Compressed Observations ===\n");
    
    // Create test observation data
    float raw_obs[] = {
        0.8f, 0.5f, 0.3f, 0.9f,  // Agent state
        0.1f, 0.2f, 0.4f, 0.6f, 0.7f, 0.2f  // Environment
    };
    int raw_size = sizeof(raw_obs) / sizeof(float);
    
    // Create compressed observation
    CompressedObs compressed;
    init_compressed_obs(&compressed);
    compress_observation(raw_obs, raw_size, &compressed);
    
    TEST_ASSERT(compressed.obs_dim > 0, "Compressed obs has positive dimension");
    TEST_ASSERT(compressed.num_rays == 128, "Correct number of rays");
    TEST_ASSERT(compressed.voxel_grid_size == 16, "Correct voxel grid size");
    TEST_ASSERT(compressed.local_grid_size == 8, "Correct local grid size");
    
    // Test decompression
    float decompressed_obs[100];
    decompress_observation(&compressed, decompressed_obs, raw_size);
    
    // Verify first few elements
    TEST_ASSERT(fabs(decompressed_obs[0] - raw_obs[0]) < 0.01f, "Decompression works for agent state");
    TEST_ASSERT(fabs(decompressed_obs[1] - raw_obs[1]) < 0.01f, "Decompression works for environment");
    
    printf("✓ Compressed observations test passed\n");
}

// Test enhanced model creation
void test_enhanced_model_creation() {
    printf("=== Testing Enhanced Model Creation ===\n");
    
    // Create enhanced configuration
    MuzeEnhancedConfig config;
    init_enhanced_config(&config);
    
    // Test discrete action configuration
    config.use_discrete_actions = 1;
    config.num_move_bins = 31;
    config.num_forward_bins = 11;
    config.num_turn_bins = 31;
    
    TEST_ASSERT(config.use_discrete_actions, "Discrete actions enabled");
    TEST_ASSERT(config.num_move_bins == 31, "Correct move bin count");
    TEST_ASSERT(config.num_forward_bins == 11, "Correct forward bin count");
    TEST_ASSERT(config.num_turn_bins == 31, "Correct turn bin count");
    
    // Test compressed observation configuration
    config.use_compressed_obs = 1;
    config.num_rays = 128;
    config.voxel_grid_size = 16;
    config.local_grid_size = 8;
    
    TEST_ASSERT(config.use_compressed_obs, "Compressed observations enabled");
    TEST_ASSERT(config.num_rays == 128, "Correct ray count");
    TEST_ASSERT(config.voxel_grid_size == 16, "Correct voxel grid size");
    TEST_ASSERT(config.local_grid_size == 8, "Correct local grid size");
    
    // Validate configuration
    TEST_ASSERT(validate_enhanced_config(&config), "Enhanced configuration is valid");
    
    printf("✓ Enhanced model creation test passed\n");
}

// Test model forward pass
void test_enhanced_model_forward() {
    printf("=== Testing Enhanced Model Forward Pass ===\n");
    
    // Create enhanced model
    MuzeEnhancedConfig config;
    init_enhanced_config(&config);
    
    MuzeEnhancedModel *model = muze_enhanced_model_create(&config);
    TEST_ASSERT(model != NULL, "Enhanced model creation failed");
    
    if (model) {
        // Create test observation
        CompressedObs obs;
        init_compressed_obs(&obs);
        
        obs.health = 0.8f;
        obs.energy = 0.6f;
        obs.inventory_size = 5;
        obs.game_time = 1000;
        
        // Forward pass
        int action = 0;
        float value = 0.0f;
        float policy = 0.0f;
        
        muze_enhanced_model_forward(model, &obs, &action, &value, &policy);
        
        TEST_ASSERT(action >= 0 && action < TOTAL_DISCRETE_ACTIONS, "Action in valid range");
        TEST_ASSERT(value >= 0.0f && value <= 1.0f, "Value in valid range");
        TEST_ASSERT(policy >= 0.0f && policy <= 1.0f, "Policy in valid range");
        
        muze_enhanced_model_destroy(model);
    }
    
    printf("✓ Enhanced model forward pass test passed\n");
}

// Test training step
void test_enhanced_training() {
    printf("=== Testing Enhanced Training ===\n");
    
    // Create enhanced model
    MuzeEnhancedConfig config;
    init_enhanced_config(&config);
    
    // Enable continuous actor for testing
    config.use_continuous_actor = 1;
    config.continuous_lr = 0.0003f;
    
    MuzeEnhancedModel *model = muze_enhanced_model_create(&config);
    TEST_ASSERT(model != NULL, "Enhanced model creation failed");
    
    if (model) {
        // Create test observation
        CompressedObs obs;
        init_compressed_obs(&obs);
        
        obs.health = 0.9f;
        obs.energy = 0.7f;
        obs.inventory_size = 3;
        obs.game_time = 2000;
        
        // Training step
        int action = 15;  // Move forward
        float reward = 0.1f;
        int done = 0;
        
        muze_enhanced_model_train_step(model, &obs, action, reward, done);
        
        TEST_ASSERT(model->training_step > 0, "Training step incremented");
        TEST_ASSERT(model->loss_value >= 0.0f, "Loss is non-negative");
        
        muze_enhanced_model_destroy(model);
    }
    
    printf("✓ Enhanced training test passed\n");
}

// Test parameter counting
void test_parameter_counting() {
    printf("=== Testing Parameter Counting ===\n");
    
    // Create enhanced model
    MuzeEnhancedConfig config;
    init_enhanced_config(&config);
    
    MuzeEnhancedModel *model = muze_enhanced_model_create(&config);
    TEST_ASSERT(model != NULL, "Enhanced model creation failed");
    
    if (model) {
        size_t param_count = muze_enhanced_model_get_parameter_count(model);
        TEST_ASSERT(param_count > 0, "Parameter count is positive");
        
        printf("  Total parameters: %zu\n", param_count);
        
        muze_enhanced_model_print_summary(model);
        
        muze_enhanced_model_destroy(model);
    }
    
    printf("✓ Parameter counting test passed\n");
}

// Main test runner
int main(void) {
    printf("=== Enhanced MUZE Framework Tests ===\n");
    
    test_discrete_actions();
    test_compressed_observations();
    test_enhanced_model_creation();
    test_enhanced_model_forward();
    test_enhanced_training();
    test_parameter_counting();
    
    printf("\n=== Enhanced MUZE Test Results ===\n");
    printf("✓ Passed: %d\n", test_passed);
    printf("✗ Failed: %d\n", test_failed);
    printf("Total: %d\n", test_passed + test_failed);
    
    if (test_failed == 0) {
        printf("\n🎉 All enhanced MUZE tests passed! Framework is ready for continuous control and compressed observations!\n");
        return 0;
    } else {
        printf("\n⚠️  Some enhanced MUZE tests failed. Please check the implementation.\n");
        return 1;
    }
}
