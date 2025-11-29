#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/stat.h>
#include "SAM/SAM.h"

#define INPUT_DIM 256
#define OUTPUT_DIM 64
#define NUM_HEADS 8
#define BATCH_SIZE 32
#define NUM_EPOCHS 100
#define LEARNING_RATE 0.001

// Generate synthetic training data
void generate_training_data(long double** inputs, long double* targets, size_t batch_size) {
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < INPUT_DIM; j++) {
            inputs[i][j] = (long double)rand() / RAND_MAX;
        }
        for (size_t j = 0; j < OUTPUT_DIM; j++) {
            targets[j] = (long double)rand() / RAND_MAX;
        }
    }
}

// Training loop for SAM
void train_sam(SAM_t* sam) {
    // Allocate memory for training data
    long double** inputs = malloc(BATCH_SIZE * sizeof(long double*));
    for (size_t i = 0; i < BATCH_SIZE; i++) {
        inputs[i] = malloc(INPUT_DIM * sizeof(long double));
    }
    long double* targets = malloc(OUTPUT_DIM * sizeof(long double));

    // Training loop
    printf("Starting training...\n");
    for (size_t epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        long double total_loss = 0.0L;

        // Process batches
        for (size_t batch = 0; batch < BATCH_SIZE; batch++) {
            // Generate synthetic data
            generate_training_data(inputs, targets, BATCH_SIZE);

            // Train SAM
            SAM_train(sam, inputs, 1, targets);  // Sequence length 1 for now

            // Calculate loss
            long double* output = SAM_forward(sam, inputs, 1);
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
        }

        // Print epoch statistics
        printf("Epoch %zu/%d - Avg Loss: %.6Lf\n", 
               epoch + 1, NUM_EPOCHS, total_loss / BATCH_SIZE);

        // Save checkpoint every 10 epochs
        if ((epoch + 1) % 10 == 0) {
            char checkpoint_file[100];
            snprintf(checkpoint_file, sizeof(checkpoint_file), 
                    "sam_checkpoint_epoch_%zu.bin", epoch + 1);
            if (SAM_save(sam, checkpoint_file) == 1) {
                printf("✓ Saved checkpoint to %s\n", checkpoint_file);
            } else {
                printf("✗ Failed to save checkpoint to %s\n", checkpoint_file);
            }
        }
    }

    // Cleanup
    for (size_t i = 0; i < BATCH_SIZE; i++) {
        free(inputs[i]);
    }
    free(inputs);
    free(targets);
}

int main(void) {
    // Seed random number generator
    srand(time(NULL));

    // Try to load existing model, otherwise initialize new one
    SAM_t* sam = SAM_load("sam_trained_model.bin");
    if (sam) {
        printf("Loaded existing model from sam_trained_model.bin\n");
        printf("Continuing training from saved model...\n\n");
    } else {
        // Initialize new SAM model
        printf("No existing model found. Initializing new SAM model...\n\n");
        sam = SAM_init(INPUT_DIM, OUTPUT_DIM, NUM_HEADS, 0);
        if (!sam) {
            fprintf(stderr, "Failed to initialize SAM\n");
            return 1;
        }
    }

    // Train SAM
    train_sam(sam);

    // Save final model after training (with timestamp)
    printf("\nSaving final trained model...\n");
    time_t now = time(NULL);
    struct tm* t = localtime(&now);
    char timestamped_file[200];
    snprintf(timestamped_file, sizeof(timestamped_file), 
             "sam_trained_%04d%02d%02d_%02d%02d%02d.bin",
             t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
             t->tm_hour, t->tm_min, t->tm_sec);
    
    // Save timestamped version
    if (SAM_save(sam, timestamped_file) == 1) {
        printf("✓ Model saved to %s\n", timestamped_file);
    } else {
        printf("✗ Failed to save model to %s\n", timestamped_file);
    }
    
    // Also save to default location for easy loading
    const char* final_model_file = "sam_trained_model.bin";
    if (SAM_save(sam, final_model_file) == 1) {
        printf("✓ Model also saved to %s (for easy loading)\n", final_model_file);
    } else {
        printf("✗ Failed to save model to %s\n", final_model_file);
    }

    // Cleanup
    SAM_destroy(sam);

    printf("\nTraining completed successfully!\n");
    return 0;
}
