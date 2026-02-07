#include "SAM/SAM.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define INPUT_DIM 512
#define OUTPUT_DIM 128
#define NUM_HEADS 12

int main(void) {
    printf("=== Stage 1 Simple Test ===\n");
    
    // Test model initialization
    printf("1. Testing model initialization...\n");
    SAM_t *sam = SAM_init(INPUT_DIM, OUTPUT_DIM, NUM_HEADS, 0);
    if (!sam) {
        printf("✗ Failed to initialize SAM\n");
        return 1;
    }
    printf("✓ SAM initialized successfully\n");
    printf("  - Input dim: %d\n", INPUT_DIM);
    printf("  - Output dim: %d\n", OUTPUT_DIM);
    printf("  - Heads: %d\n", NUM_HEADS);
    printf("  - Submodels: %zu\n", sam->num_submodels);
    
    // Test basic forward pass
    printf("\n2. Testing basic forward pass...\n");
    
    // Create simple input
    long double **inputs = malloc(sizeof(long double*));
    inputs[0] = malloc(INPUT_DIM * sizeof(long double));
    
    for (int i = 0; i < INPUT_DIM; i++) {
        inputs[0][i] = (long double)i / INPUT_DIM;
    }
    
    long double *output = SAM_forward(sam, inputs, 1);
    if (output) {
        printf("✓ Forward pass successful\n");
        printf("  - First 5 outputs: ");
        for (int i = 0; i < 5; i++) {
            printf("%.6Lf ", output[i]);
        }
        printf("\n");
        free(output);
    } else {
        printf("✗ Forward pass failed\n");
    }
    
    // Test training
    printf("\n3. Testing training...\n");
    long double *targets = malloc(OUTPUT_DIM * sizeof(long double));
    for (int i = 0; i < OUTPUT_DIM; i++) {
        targets[i] = 0.5L;
    }
    
    SAM_train(sam, inputs, 1, targets);
    printf("✓ Training completed\n");
    
    // Test adaptation
    printf("\n4. Testing adaptation...\n");
    SAM_adapt(sam, inputs, 1);
    printf("✓ Adaptation completed\n");
    
    // Test save/load
    printf("\n5. Testing save/load...\n");
    if (SAM_save(sam, "test_stage1_model.bin") == 1) {
        printf("✓ Model saved\n");
        
        SAM_t *loaded_sam = SAM_load("test_stage1_model.bin");
        if (loaded_sam) {
            printf("✓ Model loaded successfully\n");
            SAM_destroy(loaded_sam);
        } else {
            printf("✗ Model loading failed\n");
        }
    } else {
        printf("✗ Model saving failed\n");
    }
    
    // Cleanup
    free(inputs[0]);
    free(inputs);
    free(targets);
    SAM_destroy(sam);
    
    printf("\n=== Stage 1 Simple Test Completed ===\n");
    return 0;
}
