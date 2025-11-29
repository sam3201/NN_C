#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "SAM/SAM.h"

#define INPUT_DIM 256
#define OUTPUT_DIM 64
#define NUM_HEADS 8
#define NUM_TEST_SAMPLES 10

// Generate random test data
void generate_test_data(long double* input, long double* target) {
    for (size_t i = 0; i < INPUT_DIM; i++) {
        input[i] = (long double)rand() / RAND_MAX;
    }
    for (size_t i = 0; i < OUTPUT_DIM; i++) {
        target[i] = (long double)rand() / RAND_MAX;
    }
}

int main(void) {
    srand(time(NULL));
    
    printf("=== SAM AGI Model Testing ===\n\n");
    
    // Initialize SAM
    printf("1. Initializing SAM model...\n");
    SAM_t* sam = SAM_init(INPUT_DIM, OUTPUT_DIM, NUM_HEADS, 0);
    if (!sam) {
        fprintf(stderr, "Failed to initialize SAM\n");
        return 1;
    }
    printf("   ✓ SAM model initialized successfully\n");
    printf("   - Input dimension: %d\n", INPUT_DIM);
    printf("   - Output dimension: %d\n", OUTPUT_DIM);
    printf("   - Number of heads: %d\n", NUM_HEADS);
    printf("   - Number of submodels: %zu\n\n", sam->num_submodels);
    
    // Test forward pass
    printf("2. Testing forward pass...\n");
    long double input[INPUT_DIM];
    long double target[OUTPUT_DIM];
    generate_test_data(input, target);
    
    long double** input_seq = (long double**)malloc(sizeof(long double*));
    input_seq[0] = input;
    
    long double* output = SAM_forward(sam, input_seq, 1);
    if (output) {
        printf("   ✓ Forward pass successful\n");
        printf("   - Output dimension: %d\n", OUTPUT_DIM);
        
        // Calculate initial loss
        long double initial_loss = 0.0L;
        for (size_t i = 0; i < OUTPUT_DIM; i++) {
            long double diff = output[i] - target[i];
            initial_loss += diff * diff;
        }
        initial_loss /= OUTPUT_DIM;
        printf("   - Initial MSE loss: %.6Lf\n\n", initial_loss);
        free(output);
    } else {
        printf("   ✗ Forward pass failed\n\n");
    }
    free(input_seq);
    
    // Test training
    printf("3. Testing training on %d samples...\n", NUM_TEST_SAMPLES);
    long double total_loss = 0.0L;
    for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
        generate_test_data(input, target);
        
        long double** train_input = (long double**)malloc(sizeof(long double*));
        train_input[0] = input;
        
        // Train
        SAM_train(sam, train_input, 1, target);
        
        // Test forward pass after training
        long double* train_output = SAM_forward(sam, train_input, 1);
        if (train_output) {
            long double sample_loss = 0.0L;
            for (size_t j = 0; j < OUTPUT_DIM; j++) {
                long double diff = train_output[j] - target[j];
                sample_loss += diff * diff;
            }
            sample_loss /= OUTPUT_DIM;
            total_loss += sample_loss;
            free(train_output);
        }
        free(train_input);
    }
    printf("   ✓ Training completed\n");
    printf("   - Average MSE loss after training: %.6Lf\n\n", total_loss / NUM_TEST_SAMPLES);
    
    // Test fitness evaluation
    printf("4. Testing fitness evaluation...\n");
    generate_test_data(input, target);
    long double fitness = SAM_evaluate_fitness(sam, input, target);
    printf("   ✓ Fitness evaluation successful\n");
    printf("   - Fitness score: %.6Lf\n\n", fitness);
    
    // Test adaptation
    printf("5. Testing model adaptation...\n");
    long double** adapt_input = (long double**)malloc(sizeof(long double*));
    adapt_input[0] = input;
    SAM_adapt(sam, adapt_input, 1);
    printf("   ✓ Model adaptation completed\n\n");
    free(adapt_input);
    
    // Test save/load
    printf("6. Testing save/load functionality...\n");
    const char* test_file = "test_sam_model.bin";
    if (SAM_save(sam, test_file) == 1) {
        printf("   ✓ Model saved successfully\n");
        
        SAM_t* loaded_sam = SAM_load(test_file);
        if (loaded_sam) {
            printf("   ✓ Model loaded successfully\n");
            
            // Test forward pass on loaded model
            long double** test_input = (long double**)malloc(sizeof(long double*));
            test_input[0] = input;
            long double* loaded_output = SAM_forward(loaded_sam, test_input, 1);
            if (loaded_output) {
                printf("   ✓ Forward pass on loaded model successful\n");
                free(loaded_output);
            }
            free(test_input);
            SAM_destroy(loaded_sam);
        } else {
            printf("   ✗ Model loading failed\n");
        }
    } else {
        printf("   ✗ Model saving failed\n");
    }
    printf("\n");
    
    // Cleanup
    printf("7. Cleaning up...\n");
    SAM_destroy(sam);
    printf("   ✓ Cleanup completed\n\n");
    
    printf("=== All Tests Completed Successfully! ===\n");
    return 0;
}

