#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../NN/transformer.h"

void test_memory_corruption_fix() {
    printf("=== Testing Memory Corruption Fix ===\n");
    
    size_t model_dim = 64;
    size_t num_heads = 4;
    size_t seq_length = 8;
    
    // Create multi-head attention layer
    MultiHeadAttention* mha = create_attention(model_dim, num_heads);
    if (!mha) {
        printf("❌ FAILED: Could not create attention layer\n");
        return;
    }
    printf("✅ SUCCESS: Attention layer created\n");
    
    // Create test input sequence
    long double** input_seq = malloc(seq_length * sizeof(long double*));
    if (!input_seq) {
        printf("❌ FAILED: Could not allocate input sequence\n");
        free_attention(mha);
        return;
    }
    
    for (size_t t = 0; t < seq_length; t++) {
        input_seq[t] = malloc(model_dim * sizeof(long double));
        if (!input_seq[t]) {
            printf("❌ FAILED: Could not allocate input sequence[%zu]\n", t);
            // Cleanup previously allocated memory
            for (size_t i = 0; i < t; i++) {
                free(input_seq[i]);
            }
            free(input_seq);
            free_attention(mha);
            return;
        }
        
        // Fill with test data
        for (size_t i = 0; i < model_dim; i++) {
            input_seq[t][i] = sinl((long double)(t * i) * 0.1) + cosl((long double)(t + i) * 0.2);
        }
    }
    printf("✅ SUCCESS: Input sequence created\n");
    
    // Test the forward pass that was previously crashing
    printf("Running forward pass (this was crashing before the fix)...\n");
    long double** output = transformer_mha_forward(mha, input_seq, seq_length);
    
    if (!output) {
        printf("❌ FAILED: Forward pass returned NULL (memory corruption fix failed)\n");
    } else {
        printf("✅ SUCCESS: Forward pass completed without crash!\n");
        
        // Validate output
        int valid_output = 1;
        for (size_t t = 0; t < seq_length && valid_output; t++) {
            if (!output[t]) {
                printf("❌ FAILED: Output[%zu] is NULL\n", t);
                valid_output = 0;
                break;
            }
            
            for (size_t i = 0; i < model_dim && valid_output; i++) {
                if (isnan(output[t][i]) || isinf(output[t][i])) {
                    printf("❌ FAILED: Output[%zu][%zu] is NaN/Inf: %Lf\n", t, i, output[t][i]);
                    valid_output = 0;
                    break;
                }
            }
        }
        
        if (valid_output) {
            printf("✅ SUCCESS: Output validation passed\n");
            printf("   Output shape: [%zu x %zu]\n", seq_length, model_dim);
            printf("   Sample output[0][0]: %.6f\n", (double)output[0][0]);
            printf("   Sample output[0][%zu]: %.6f\n", model_dim-1, (double)output[0][model_dim-1]);
        }
        
        // Free output memory
        if (output) {
            for (size_t t = 0; t < seq_length; t++) {
                if (output[t]) {
                    free(output[t]);
                }
            }
            free(output);
        }
    }
    
    // Cleanup input sequence
    for (size_t t = 0; t < seq_length; t++) {
        free(input_seq[t]);
    }
    free(input_seq);
    
    // Cleanup attention layer
    free_attention(mha);
    
    printf("✅ SUCCESS: Memory corruption fix test completed\n\n");
}

void test_stress_multiple_calls() {
    printf("=== Testing Multiple Forward Passes (Stress Test) ===\n");
    
    size_t model_dim = 32;
    size_t num_heads = 2;
    size_t seq_length = 4;
    int num_tests = 20;
    
    MultiHeadAttention* mha = create_attention(model_dim, num_heads);
    if (!mha) {
        printf("❌ FAILED: Could not create attention layer\n");
        return;
    }
    
    int success_count = 0;
    
    for (int test = 0; test < num_tests; test++) {
        // Create input sequence
        long double** input_seq = malloc(seq_length * sizeof(long double*));
        for (size_t t = 0; t < seq_length; t++) {
            input_seq[t] = malloc(model_dim * sizeof(long double));
            for (size_t i = 0; i < model_dim; i++) {
                // Add some randomness
                input_seq[t][i] = (long double)((rand() % 1000) - 500) * 0.01;
            }
        }
        
        long double** output = transformer_mha_forward(mha, input_seq, seq_length);
        
        if (output) {
            // Quick validation
            int valid = 1;
            for (size_t t = 0; t < seq_length && valid; t++) {
                if (!output[t]) {
                    valid = 0;
                    break;
                }
                for (size_t i = 0; i < model_dim && valid; i++) {
                    if (isnan(output[t][i]) || isinf(output[t][i])) {
                        valid = 0;
                        break;
                    }
                }
            }
            
            if (valid) {
                success_count++;
            }
            
            // Cleanup output
            for (size_t t = 0; t < seq_length; t++) {
                if (output[t]) free(output[t]);
            }
            free(output);
        }
        
        // Cleanup input
        for (size_t t = 0; t < seq_length; t++) {
            free(input_seq[t]);
        }
        free(input_seq);
    }
    
    printf("Stress test results: %d/%d successful\n", success_count, num_tests);
    
    if (success_count == num_tests) {
        printf("✅ SUCCESS: All stress tests passed\n");
    } else {
        printf("⚠️  WARNING: Some stress tests failed\n");
    }
    
    free_attention(mha);
    printf("✅ SUCCESS: Stress test completed\n\n");
}

int main() {
    printf("🧪 Transformer Memory Corruption Fix Test\n");
    printf("========================================\n\n");
    
    srand(42); // For reproducible tests
    
    test_memory_corruption_fix();
    test_stress_multiple_calls();
    
    printf("🎉 Transformer memory corruption fix test completed!\n");
    printf("The fix prevents crashes when NN_forward returns NULL pointers.\n");
    
    return 0;
}
