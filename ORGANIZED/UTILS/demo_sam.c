#include "SAM/SAM.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INPUT_DIM 256
#define OUTPUT_DIM 64
#define NUM_HEADS 8

// Generate test input patterns
void generate_pattern_input(long double *input, int pattern_type) {
    switch (pattern_type) {
        case 0: // Sine wave pattern
            for (size_t i = 0; i < INPUT_DIM; i++) {
                input[i] = sin(i * 0.1) * 0.8 + 0.1;
            }
            break;
        case 1: // Cosine wave pattern
            for (size_t i = 0; i < INPUT_DIM; i++) {
                input[i] = cos(i * 0.15) * 0.7 + 0.15;
            }
            break;
        case 2: // Random noise pattern
            for (size_t i = 0; i < INPUT_DIM; i++) {
                input[i] = (long double)rand() / RAND_MAX;
            }
            break;
        case 3: // Linear pattern
            for (size_t i = 0; i < INPUT_DIM; i++) {
                input[i] = (long double)i / INPUT_DIM;
            }
            break;
        default:
            for (size_t i = 0; i < INPUT_DIM; i++) {
                input[i] = 0.5;
            }
    }
}

// Test model with different patterns
void test_pattern_recognition(SAM_t *sam) {
    printf("\n=== Pattern Recognition Tests ===\n");
    
    const char *pattern_names[] = {"Sine Wave", "Cosine Wave", "Random Noise", "Linear"};
    
    for (int pattern = 0; pattern < 4; pattern++) {
        printf("\nTesting %s Pattern:\n", pattern_names[pattern]);
        
        // Generate input
        long double input[INPUT_DIM];
        generate_pattern_input(input, pattern);
        
        // Create input sequence
        long double **input_seq = malloc(sizeof(long double*));
        input_seq[0] = input;
        
        // Get model output
        long double *output = SAM_forward(sam, input_seq, 1);
        if (output) {
            // Calculate output statistics
            long double sum = 0.0L, min_val = output[0], max_val = output[0];
            for (size_t i = 0; i < OUTPUT_DIM; i++) {
                sum += output[i];
                if (output[i] < min_val) min_val = output[i];
                if (output[i] > max_val) max_val = output[i];
            }
            long double mean = sum / OUTPUT_DIM;
            
            printf("  ✓ Output generated successfully\n");
            printf("  - Mean: %.6Lf\n", mean);
            printf("  - Range: [%.6Lf, %.6Lf]\n", min_val, max_val);
            printf("  - First 5 values: ");
            for (int i = 0; i < 5; i++) {
                printf("%.6Lf ", output[i]);
            }
            printf("\n");
            
            free(output);
        } else {
            printf("  ✗ Failed to generate output\n");
        }
        
        free(input_seq);
    }
}

// Test model adaptation
void test_adaptation(SAM_t *sam) {
    printf("\n=== Adaptation Test ===\n");
    
    // Create a simple learning scenario
    long double input[INPUT_DIM];
    long double target[OUTPUT_DIM];
    
    // Initialize with a pattern
    for (size_t i = 0; i < INPUT_DIM; i++) {
        input[i] = sin(i * 0.1);
    }
    for (size_t i = 0; i < OUTPUT_DIM; i++) {
        target[i] = cos(i * 0.2);
    }
    
    long double **input_seq = malloc(sizeof(long double*));
    input_seq[0] = input;
    
    // Test before adaptation
    printf("Testing before adaptation...\n");
    long double *output_before = SAM_forward(sam, input_seq, 1);
    if (output_before) {
        long double loss_before = 0.0L;
        for (size_t i = 0; i < OUTPUT_DIM; i++) {
            long double error = output_before[i] - target[i];
            loss_before += error * error;
        }
        printf("  Initial MSE loss: %.6Lf\n", loss_before / OUTPUT_DIM);
        free(output_before);
    }
    
    // Adapt the model
    printf("Adapting model...\n");
    for (int i = 0; i < 5; i++) {
        SAM_train(sam, input_seq, 1, target);
        SAM_adapt(sam, input_seq, 1);
    }
    
    // Test after adaptation
    printf("Testing after adaptation...\n");
    long double *output_after = SAM_forward(sam, input_seq, 1);
    if (output_after) {
        long double loss_after = 0.0L;
        for (size_t i = 0; i < OUTPUT_DIM; i++) {
            long double error = output_after[i] - target[i];
            loss_after += error * error;
        }
        printf("  Final MSE loss: %.6Lf\n", loss_after / OUTPUT_DIM);
        free(output_after);
    }
    
    free(input_seq);
}

// Test model fitness
void test_fitness_evaluation(SAM_t *sam) {
    printf("\n=== Fitness Evaluation ===\n");
    
    long double input[INPUT_DIM];
    long double target[OUTPUT_DIM];
    
    // Generate test data
    for (size_t i = 0; i < INPUT_DIM; i++) {
        input[i] = (long double)rand() / RAND_MAX;
    }
    for (size_t i = 0; i < OUTPUT_DIM; i++) {
        target[i] = (long double)rand() / RAND_MAX;
    }
    
    long double fitness = SAM_evaluate_fitness(sam, input, target);
    printf("Fitness score: %.6Lf\n", fitness);
    printf("(Higher fitness indicates better performance)\n");
}

int main(void) {
    printf("=== SAM AGI Model Demonstration ===\n");
    
    srand(time(NULL));
    
    // Load the trained model
    printf("Loading trained SAM model...\n");
    SAM_t *sam = SAM_load("sam_production_model.bin");
    if (!sam) {
        printf("Failed to load production model, trying debug model...\n");
        sam = SAM_load("debug_sam_model.bin");
    }
    
    if (!sam) {
        fprintf(stderr, "Failed to load any trained model\n");
        return 1;
    }
    
    printf("✓ Model loaded successfully\n");
    printf("  - Input dimension: %d\n", INPUT_DIM);
    printf("  - Output dimension: %d\n", OUTPUT_DIM);
    printf("  - Number of heads: %d\n", NUM_HEADS);
    printf("  - Number of submodels: %zu\n\n", sam->num_submodels);
    
    // Run tests
    test_pattern_recognition(sam);
    test_adaptation(sam);
    test_fitness_evaluation(sam);
    
    printf("\n=== Demonstration Completed ===\n");
    printf("The SAM AGI model is ready for integration!\n");
    
    // Cleanup
    SAM_destroy(sam);
    
    return 0;
}
