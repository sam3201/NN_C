#include "SAM/SAM.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INPUT_DIM 256
#define OUTPUT_DIM 64
#define NUM_HEADS 8
#define BATCH_SIZE 4
#define NUM_EPOCHS 5

int main(void) {
    printf("=== Debug SAM Training ===\n");
    
    srand(time(NULL));
    
    // Initialize SAM model
    printf("1. Initializing SAM model...\n");
    SAM_t *sam = SAM_init(INPUT_DIM, OUTPUT_DIM, NUM_HEADS, 0);
    if (!sam) {
        fprintf(stderr, "Failed to initialize SAM\n");
        return 1;
    }
    printf("   ✓ SAM model initialized successfully\n");
    printf("   - Number of submodels: %zu\n\n", sam->num_submodels);
    
    // Simple training test
    printf("2. Starting simple training test...\n");
    for (size_t epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        printf("   Epoch %zu/%d\n", epoch + 1, NUM_EPOCHS);
        
        // Create simple input data
        long double **inputs = malloc(sizeof(long double*));
        inputs[0] = malloc(INPUT_DIM * sizeof(long double));
        long double *targets = malloc(OUTPUT_DIM * sizeof(long double));
        
        // Generate simple data
        for (size_t i = 0; i < INPUT_DIM; i++) {
            inputs[0][i] = 0.5L;
        }
        for (size_t i = 0; i < OUTPUT_DIM; i++) {
            targets[i] = 0.8L;
        }
        
        // Train
        printf("     Training...");
        SAM_train(sam, inputs, 1, targets);
        printf(" Done\n");
        
        // Test forward pass
        printf("     Forward pass...");
        long double *output = SAM_forward(sam, inputs, 1);
        if (output) {
            printf(" Success\n");
            free(output);
        } else {
            printf(" Failed\n");
        }
        
        // Adapt
        printf("     Adapting...");
        SAM_adapt(sam, inputs, 1);
        printf(" Done\n");
        
        // Cleanup
        free(inputs[0]);
        free(inputs);
        free(targets);
    }
    
    printf("   ✓ Training completed successfully\n\n");
    
    // Save model
    printf("3. Saving model...\n");
    if (SAM_save(sam, "debug_sam_model.bin") == 1) {
        printf("   ✓ Model saved to debug_sam_model.bin\n");
    } else {
        printf("   ✗ Failed to save model\n");
    }
    
    // Cleanup
    printf("4. Cleaning up...\n");
    SAM_destroy(sam);
    printf("   ✓ Cleanup completed\n\n");
    
    printf("=== Debug Training Completed Successfully! ===\n");
    return 0;
}
