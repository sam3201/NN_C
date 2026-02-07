#include "SAM/SAM.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INPUT_DIM 256
#define OUTPUT_DIM 64
#define NUM_HEADS 8
#define BATCH_SIZE 8
#define NUM_EPOCHS 20
#define LEARNING_RATE 0.001

// Generate realistic training data
void generate_training_data(long double **inputs, long double *targets,
                            size_t batch_size) {
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < INPUT_DIM; j++) {
            // Generate more realistic input patterns
            inputs[i][j] = sin(j * 0.1) * cos(i * 0.2) + (long double)rand() / RAND_MAX * 0.1;
        }
        for (size_t j = 0; j < OUTPUT_DIM; j++) {
            // Generate corresponding targets with some pattern
            targets[j] = cos(j * 0.15) * sin(i * 0.1) + (long double)rand() / RAND_MAX * 0.1;
        }
    }
}

// Training loop for SAM
void train_sam(SAM_t *sam) {
    printf("Starting production training...\n");
    
    for (size_t epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        long double total_loss = 0.0L;
        
        // Process batches
        for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
            // Allocate memory for this batch
            long double **inputs = malloc(sizeof(long double*));
            inputs[0] = malloc(INPUT_DIM * sizeof(long double));
            long double *targets = malloc(OUTPUT_DIM * sizeof(long double));
            
            // Generate training data
            generate_training_data(inputs, targets, 1);
            
            // Train SAM
            SAM_train(sam, inputs, 1, targets);
            
            // Calculate loss
            long double *output = SAM_forward(sam, inputs, 1);
            if (output) {
                long double batch_loss = 0.0L;
                for (size_t i = 0; i < OUTPUT_DIM; i++) {
                    long double error = output[i] - targets[i];
                    batch_loss += error * error;
                }
                total_loss += batch_loss / OUTPUT_DIM;
                free(output);
            }
            
            // Adapt models
            SAM_adapt(sam, inputs, 1);
            
            // Cleanup
            free(inputs[0]);
            free(inputs);
            free(targets);
        }
        
        // Print epoch statistics
        printf("Epoch %zu/%d - Avg Loss: %.6Lf\n", epoch + 1, NUM_EPOCHS,
               total_loss / BATCH_SIZE);
        
        // Save checkpoint every 5 epochs
        if ((epoch + 1) % 5 == 0) {
            char checkpoint_file[100];
            snprintf(checkpoint_file, sizeof(checkpoint_file),
                     "sam_checkpoint_epoch_%zu.bin", epoch + 1);
            if (SAM_save(sam, checkpoint_file) == 1) {
                printf("  ✓ Saved checkpoint to %s\n", checkpoint_file);
            } else {
                printf("  ✗ Failed to save checkpoint to %s\n", checkpoint_file);
            }
        }
    }
}

int main(void) {
    printf("=== SAM AGI Production Training ===\n\n");
    
    // Seed random number generator
    srand(time(NULL));
    
    // Try to load existing model, otherwise initialize new one
    SAM_t *sam = SAM_load("sam_production_model.bin");
    if (sam) {
        printf("Loaded existing model from sam_production_model.bin\n");
        printf("Continuing training from saved model...\n\n");
    } else {
        // Initialize new SAM model
        printf("No existing model found. Initializing new SAM model...\n\n");
        sam = SAM_init(INPUT_DIM, OUTPUT_DIM, NUM_HEADS, 0);
        if (!sam) {
            fprintf(stderr, "Failed to initialize SAM\n");
            return 1;
        }
        printf("✓ SAM model initialized successfully\n");
        printf("  - Input dimension: %d\n", INPUT_DIM);
        printf("  - Output dimension: %d\n", OUTPUT_DIM);
        printf("  - Number of heads: %d\n", NUM_HEADS);
        printf("  - Number of submodels: %zu\n\n", sam->num_submodels);
    }
    
    // Train SAM
    train_sam(sam);
    
    // Save final model after training (with timestamp)
    printf("\nSaving final trained model...\n");
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    char timestamped_file[200];
    snprintf(timestamped_file, sizeof(timestamped_file),
             "sam_production_%04d%02d%02d_%02d%02d%02d.bin", t->tm_year + 1900,
             t->tm_mon + 1, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);
    
    // Save timestamped version
    if (SAM_save(sam, timestamped_file) == 1) {
        printf("✓ Model saved to %s\n", timestamped_file);
    } else {
        printf("✗ Failed to save model to %s\n", timestamped_file);
    }
    
    // Also save to default location for easy loading
    const char *final_model_file = "sam_production_model.bin";
    if (SAM_save(sam, final_model_file) == 1) {
        printf("✓ Model also saved to %s (for easy loading)\n", final_model_file);
    } else {
        printf("✗ Failed to save model to %s\n", final_model_file);
    }
    
    // Cleanup
    SAM_destroy(sam);
    
    printf("\n=== Production Training Completed Successfully! ===\n");
    return 0;
}
