#include "../NN/RNN/RNN.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

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

#define TEST_ASSERT_CLOSE(a, b, tolerance, message) \
    TEST_ASSERT(fabs(a - b) < tolerance, message)

// Test data
static long double rnn_inputs[] = {
    0.1L, 0.2L, 0.3L, 0.4L,  // Time step 1
    0.2L, 0.3L, 0.4L, 0.5L,  // Time step 2
    0.3L, 0.4L, 0.5L, 0.6L,  // Time step 3
    0.4L, 0.5L, 0.6L, 0.7L   // Time step 4
};

static long double rnn_targets[] = {
    0.5L, 0.6L, 0.7L, 0.8L   // Expected output
};

// Test RNN creation
void test_rnn_creation() {
    printf("=== Testing RNN Creation ===\n");
    
    // Test simple RNN
    RNN_t *simple_rnn = RNN_create(4, 8, 4, 2, RNN_SIMPLE);
    TEST_ASSERT(simple_rnn != NULL, "Simple RNN creation failed");
    
    if (simple_rnn) {
        TEST_ASSERT(simple_rnn->num_layers == 2, "Simple RNN layer count incorrect");
        TEST_ASSERT(simple_rnn->input_size == 4, "Simple RNN input size incorrect");
        TEST_ASSERT(simple_rnn->hidden_size == 8, "Simple RNN hidden size incorrect");
        TEST_ASSERT(simple_rnn->output_size == 4, "Simple RNN output size incorrect");
        
        RNN_destroy(simple_rnn);
    }
    
    // Test LSTM
    RNN_t *lstm = RNN_create(4, 8, 4, 2, RNN_LSTM);
    TEST_ASSERT(lstm != NULL, "LSTM creation failed");
    
    if (lstm) {
        TEST_ASSERT(lstm->num_layers == 2, "LSTM layer count incorrect");
        TEST_ASSERT(lstm->input_size == 4, "LSTM input size incorrect");
        TEST_ASSERT(lstm->hidden_size == 8, "LSTM hidden size incorrect");
        TEST_ASSERT(lstm->output_size == 4, "LSTM output size incorrect");
        
        RNN_destroy(lstm);
    }
    
    printf("✓ RNN creation test passed\n");
}

// Test RNN forward pass
void test_rnn_forward() {
    printf("=== Testing RNN Forward Pass ===\n");
    
    RNN_t *rnn = RNN_create(4, 8, 4, 2, RNN_SIMPLE);
    TEST_ASSERT(rnn != NULL, "RNN creation failed");
    
    if (rnn) {
        // Test single step forward pass
        long double *output = RNN_forward(rnn, rnn_inputs, 1);
        TEST_ASSERT(output != NULL, "RNN forward pass failed");
        
        if (output) {
            // Check output is reasonable (between 0 and 1 for sigmoid)
            for (size_t i = 0; i < 4; i++) {
                TEST_ASSERT(output[i] >= 0.0L && output[i] <= 1.0L, "RNN output out of range");
            }
            free(output);
        }
        
        // Test sequence forward pass
        long double *seq_output = RNN_forward(rnn, rnn_inputs, 4);
        TEST_ASSERT(seq_output != NULL, "RNN sequence forward pass failed");
        
        if (seq_output) {
            // Check output is reasonable
            for (size_t i = 0; i < 4; i++) {
                TEST_ASSERT(seq_output[i] >= 0.0L && seq_output[i] <= 1.0L, "RNN sequence output out of range");
            }
            free(seq_output);
        }
        
        RNN_destroy(rnn);
    }
    
    printf("✓ RNN forward pass test passed\n");
}

// Test LSTM forward pass
void test_lstm_forward() {
    printf("=== Testing LSTM Forward Pass ===\n");
    
    RNN_t *lstm = RNN_create(4, 8, 4, 2, RNN_LSTM);
    TEST_ASSERT(lstm != NULL, "LSTM creation failed");
    
    if (lstm) {
        // Test single step forward pass
        long double *output = RNN_forward(lstm, rnn_inputs, 1);
        TEST_ASSERT(output != NULL, "LSTM forward pass failed");
        
        if (output) {
            // Check output is reasonable (between 0 and 1 for sigmoid)
            for (size_t i = 0; i < 4; i++) {
                TEST_ASSERT(output[i] >= 0.0L && output[i] <= 1.0L, "LSTM output out of range");
            }
            free(output);
        }
        
        // Test sequence forward pass
        long double *seq_output = RNN_forward(lstm, rnn_inputs, 4);
        TEST_ASSERT(seq_output != NULL, "LSTM sequence forward pass failed");
        
        if (seq_output) {
            // Check output is reasonable
            for (size_t i = 0; i < 4; i++) {
                TEST_ASSERT(seq_output[i] >= 0.0L && seq_output[i] <= 1.0L, "LSTM sequence output out of range");
            }
            free(seq_output);
        }
        
        RNN_destroy(lstm);
    }
    
    printf("✓ LSTM forward pass test passed\n");
}

// Test RNN state management
void test_rnn_states() {
    printf("=== Testing RNN State Management ===\n");
    
    RNN_t *rnn = RNN_create(4, 8, 4, 2, RNN_SIMPLE);
    TEST_ASSERT(rnn != NULL, "RNN creation failed");
    
    if (rnn) {
        // Test state reset
        RNN_reset_states(rnn);
        
        // Forward pass should work after reset
        long double *output1 = RNN_forward(rnn, rnn_inputs, 1);
        TEST_ASSERT(output1 != NULL, "RNN forward pass after reset failed");
        
        // Second forward pass should give different results
        long double *output2 = RNN_forward(rnn, rnn_inputs, 1);
        TEST_ASSERT(output2 != NULL, "RNN second forward pass failed");
        
        if (output1 && output2) {
            // Outputs should be different (due to hidden state)
            bool different = false;
            for (size_t i = 0; i < 4; i++) {
                if (fabsl(output1[i] - output2[i]) > 1e-6L) {
                    different = true;
                    break;
                }
            }
            TEST_ASSERT(different, "RNN outputs should be different across timesteps");
            
            free(output1);
            free(output2);
        }
        
        // Reset and test again
        RNN_reset_states(rnn);
        long double *output3 = RNN_forward(rnn, rnn_inputs, 1);
        TEST_ASSERT(output3 != NULL, "RNN forward pass after second reset failed");
        
        if (output3) {
            free(output3);
        }
        
        RNN_destroy(rnn);
    }
    
    printf("✓ RNN state management test passed\n");
}

// Test LSTM state management
void test_lstm_states() {
    printf("=== Testing LSTM State Management ===\n");
    
    RNN_t *lstm = RNN_create(4, 8, 4, 2, RNN_LSTM);
    TEST_ASSERT(lstm != NULL, "LSTM creation failed");
    
    if (lstm) {
        // Test state reset
        RNN_reset_states(lstm);
        
        // Forward pass should work after reset
        long double *output1 = RNN_forward(lstm, rnn_inputs, 1);
        TEST_ASSERT(output1 != NULL, "LSTM forward pass after reset failed");
        
        // Second forward pass should give different results
        long double *output2 = RNN_forward(lstm, rnn_inputs, 1);
        TEST_ASSERT(output2 != NULL, "LSTM second forward pass failed");
        
        if (output1 && output2) {
            // Outputs should be different (due to hidden and cell states)
            bool different = false;
            for (size_t i = 0; i < 4; i++) {
                if (fabsl(output1[i] - output2[i]) > 1e-6L) {
                    different = true;
                    break;
                }
            }
            TEST_ASSERT(different, "LSTM outputs should be different across timesteps");
            
            free(output1);
            free(output2);
        }
        
        // Reset and test again
        RNN_reset_states(lstm);
        long double *output3 = RNN_forward(lstm, rnn_inputs, 1);
        TEST_ASSERT(output3 != NULL, "LSTM forward pass after second reset failed");
        
        if (output3) {
            free(output3);
        }
        
        RNN_destroy(lstm);
    }
    
    printf("✓ LSTM state management test passed\n");
}

// Test RNN memory management
void test_rnn_memory() {
    printf("=== Testing RNN Memory Management ===\n");
    
    // Test multiple creation and destruction
    for (int i = 0; i < 5; i++) {
        RNN_t *rnn = RNN_create(4, 8, 4, 2, RNN_SIMPLE);
        TEST_ASSERT(rnn != NULL, "RNN creation failed");
        
        if (rnn) {
            long double *output = RNN_forward(rnn, rnn_inputs, 1);
            if (output) free(output);
            
            RNN_destroy(rnn);
        }
        
        printf("  Memory test %d passed\n", i + 1);
    }
    
    // Test LSTM memory management
    for (int i = 0; i < 5; i++) {
        RNN_t *lstm = RNN_create(4, 8, 4, 2, RNN_LSTM);
        TEST_ASSERT(lstm != NULL, "LSTM creation failed");
        
        if (lstm) {
            long double *output = RNN_forward(lstm, rnn_inputs, 1);
            if (output) free(output);
            
            RNN_destroy(lstm);
        }
        
        printf("  LSTM memory test %d passed\n", i + 1);
    }
    
    printf("✓ RNN memory management test passed\n");
}

// Test RNN parameter count
void test_rnn_parameters() {
    printf("=== Testing RNN Parameter Count ===\n");
    
    RNN_t *rnn = RNN_create(4, 8, 4, 2, RNN_SIMPLE);
    TEST_ASSERT(rnn != NULL, "RNN creation failed");
    
    if (rnn) {
        size_t param_count = RNN_get_parameter_count(rnn);
        TEST_ASSERT(param_count > 0, "RNN parameter count should be positive");
        
        // Simple RNN: 2 layers * (W_xh + W_hh + W_hy + b_h + b_y)
        // = 2 * (4*8 + 8*8 + 8*4 + 8 + 4) = 2 * (32 + 64 + 32 + 8 + 4) = 2 * 140 = 280
        size_t expected_params = 280;
        TEST_ASSERT(param_count == expected_params, "RNN parameter count incorrect");
        
        printf("  RNN parameters: %zu (expected: %zu)\n", param_count, expected_params);
        
        RNN_destroy(rnn);
    }
    
    // Test LSTM parameter count
    RNN_t *lstm = RNN_create(4, 8, 4, 2, RNN_LSTM);
    TEST_ASSERT(lstm != NULL, "LSTM creation failed");
    
    if (lstm) {
        size_t param_count = RNN_get_parameter_count(lstm);
        TEST_ASSERT(param_count > 0, "LSTM parameter count should be positive");
        
        // LSTM has more parameters due to gates
        printf("  LSTM parameters: %zu\n", param_count);
        
        RNN_destroy(lstm);
    }
    
    printf("✓ RNN parameter count test passed\n");
}

// Test RNN utilities
void test_rnn_utilities() {
    printf("=== Testing RNN Utilities ===\n");
    
    RNN_t *rnn = RNN_create(4, 8, 4, 2, RNN_SIMPLE);
    TEST_ASSERT(rnn != NULL, "RNN creation failed");
    
    if (rnn) {
        // Test summary
        printf("  RNN Summary:\n");
        RNN_print_summary(rnn);
        
        // Test sequence length setting
        RNN_set_sequence_length(rnn, 10);
        TEST_ASSERT(rnn->sequence_length == 10, "RNN sequence length setting failed");
        
        RNN_destroy(rnn);
    }
    
    printf("✓ RNN utilities test passed\n");
}

// Test game-specific RNN functions
void test_game_rnn_functions() {
    printf("=== Testing Game-Specific RNN Functions ===\n");
    
    // Test game network creation
    RNN_t *game_rnn = RNN_create_game_network(10, 16, 8);
    TEST_ASSERT(game_rnn != NULL, "Game RNN creation failed");
    
    if (game_rnn) {
        // Test game state prediction
        long double game_state[] = {0.1L, 0.2L, 0.3L, 0.4L, 0.5L, 0.6L, 0.7L, 0.8L, 0.9L, 1.0L};
        long double *prediction = RNN_predict_game_state(game_rnn, game_state);
        TEST_ASSERT(prediction != NULL, "Game state prediction failed");
        
        if (prediction) {
            // Check prediction is reasonable
            for (size_t i = 0; i < 8; i++) {
                TEST_ASSERT(prediction[i] >= 0.0L && prediction[i] <= 1.0L, "Game prediction out of range");
            }
            free(prediction);
        }
        
        // Test game sequence training
        long double sequence[] = {
            0.1L, 0.2L, 0.3L, 0.4L, 0.5L, 0.6L, 0.7L, 0.8L, 0.9L, 1.0L,  // Step 1
            0.2L, 0.3L, 0.4L, 0.5L, 0.6L, 0.7L, 0.8L, 0.9L, 1.0L, 1.1L,  // Step 2
            0.3L, 0.4L, 0.5L, 0.6L, 0.7L, 0.8L, 0.9L, 1.0L, 1.1L, 1.2L   // Step 3
        };
        long double targets[] = {0.5L, 0.6L, 0.7L, 0.8L, 0.9L, 1.0L, 1.1L, 1.2L};
        
        RNN_train_game_sequence(game_rnn, sequence, targets, 3);
        
        RNN_destroy(game_rnn);
    }
    
    printf("✓ Game-specific RNN functions test passed\n");
}

// Main test runner
int main(void) {
    printf("=== RNN/LSTM Framework Tests ===\n");
    
    test_rnn_creation();
    test_rnn_forward();
    test_lstm_forward();
    test_rnn_states();
    test_lstm_states();
    test_rnn_memory();
    test_rnn_parameters();
    test_rnn_utilities();
    test_game_rnn_functions();
    
    printf("\n=== RNN/LSTM Test Results ===\n");
    printf("✓ Passed: %d\n", test_passed);
    printf("✗ Failed: %d\n", test_failed);
    printf("Total: %d\n", test_passed + test_failed);
    
    if (test_failed == 0) {
        printf("\n🎉 All RNN/LSTM tests passed! Framework is working correctly.\n");
        return 0;
    } else {
        printf("\n⚠️  Some RNN/LSTM tests failed. Please check the implementation.\n");
        return 1;
    }
}
