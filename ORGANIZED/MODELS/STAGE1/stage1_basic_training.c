#include "SAM/SAM.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define INPUT_DIM 256
#define OUTPUT_DIM 64
#define NUM_HEADS 8
#define EPOCHS 10
#define SAMPLES_PER_EPOCH 5

typedef struct {
    long double total_loss;
    size_t samples_processed;
    time_t start_time;
} TrainingStats;

// Initialize model
SAM_t* init_training_model() {
    printf("=== SAM AGI Stage 1: Basic Raw Training ===\n\n");
    printf("Initializing training model...\n");
    printf("  Input: %d, Output: %d, Heads: %d\n", INPUT_DIM, OUTPUT_DIM, NUM_HEADS);
    
    SAM_t *sam = SAM_init(INPUT_DIM, OUTPUT_DIM, NUM_HEADS, 0);
    if (!sam) {
        fprintf(stderr, "Failed to initialize SAM\n");
        return NULL;
    }
    
    printf("✓ Model initialized with %zu submodels\n\n", sam->num_submodels);
    return sam;
}

// Load text content
char* load_text_content(const char *filename, size_t *size) {
    FILE *file = fopen(filename, "r");
    if (!file) return NULL;
    
    fseek(file, 0, SEEK_END);
    *size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    char *content = malloc(*size + 1);
    if (content) {
        fread(content, 1, *size, file);
        content[*size] = '\0';
    }
    
    fclose(file);
    return content;
}

// Create training sample from text
void create_training_sample(const char *text, size_t text_size, 
                           long double *input, long double *target) {
    // Pick random starting position
    size_t pos = rand() % (text_size - INPUT_DIM - OUTPUT_DIM);
    
    // Create input vector
    for (int i = 0; i < INPUT_DIM; i++) {
        if (pos + i < text_size) {
            input[i] = (long double)text[pos + i] / 256.0L;
        } else {
            input[i] = 0.0L;
        }
    }
    
    // Create target vector (next characters)
    for (int i = 0; i < OUTPUT_DIM; i++) {
        if (pos + INPUT_DIM + i < text_size) {
            target[i] = (long double)text[pos + INPUT_DIM + i] / 256.0L;
        } else {
            target[i] = 0.0L;
        }
    }
}

// Train one sample
void train_sample(SAM_t *sam, const char *text, size_t text_size, TrainingStats *stats) {
    // Allocate memory
    long double *input = malloc(INPUT_DIM * sizeof(long double));
    long double *target = malloc(OUTPUT_DIM * sizeof(long double));
    
    if (!input || !target) {
        free(input);
        free(target);
        return;
    }
    
    // Create training sample
    create_training_sample(text, text_size, input, target);
    
    // Create input sequence for SAM
    long double **input_seq = malloc(sizeof(long double*));
    input_seq[0] = input;
    
    // Train the model
    SAM_train(sam, input_seq, 1, target);
    
    // Calculate loss
    long double *output = SAM_forward(sam, input_seq, 1);
    if (output) {
        long double sample_loss = 0.0L;
        for (int i = 0; i < OUTPUT_DIM; i++) {
            long double error = output[i] - target[i];
            sample_loss += error * error;
        }
        stats->total_loss += sample_loss / OUTPUT_DIM;
        stats->samples_processed++;
        free(output);
    }
    
    // Adapt the model
    SAM_adapt(sam, input_seq, 1);
    
    // Cleanup
    free(input_seq);
    free(input);
    free(target);
}

// Simple test of model capabilities
void test_model_basic(SAM_t *sam) {
    printf("\n=== Basic Model Test ===\n");
    
    // Create simple test input
    long double *test_input = malloc(INPUT_DIM * sizeof(long double));
    long double **input_seq = malloc(sizeof(long double*));
    input_seq[0] = test_input;
    
    // Fill with pattern
    for (int i = 0; i < INPUT_DIM; i++) {
        test_input[i] = (long double)(i % 256) / 256.0L;
    }
    
    // Get output
    long double *output = SAM_forward(sam, input_seq, 1);
    if (output) {
        printf("✓ Forward pass successful\n");
        printf("  First 5 outputs: ");
        for (int i = 0; i < 5; i++) {
            printf("%.3Lf ", output[i]);
        }
        printf("\n");
        free(output);
    } else {
        printf("✗ Forward pass failed\n");
    }
    
    free(input_seq);
    free(test_input);
}

int main(int argc, char *argv[]) {
    srand(time(NULL));
    
    // Initialize model
    SAM_t *sam = init_training_model();
    if (!sam) return 1;
    
    // Load training data
    const char *filename = "training_data/raw_texts/Frankenstein.txt";
    if (argc > 1) filename = argv[1];
    
    printf("Loading training data: %s\n", filename);
    size_t text_size;
    char *text_content = load_text_content(filename, &text_size);
    
    if (!text_content) {
        printf("Failed to load training data\n");
        SAM_destroy(sam);
        return 1;
    }
    
    printf("✓ Loaded %zu characters of training data\n\n", text_size);
    
    // Initialize training stats
    TrainingStats stats = {0};
    stats.start_time = time(NULL);
    
    printf("=== Starting Raw Pattern Training ===\n");
    printf("Training configuration:\n");
    printf("  Epochs: %d\n", EPOCHS);
    printf("  Samples per epoch: %d\n", SAMPLES_PER_EPOCH);
    printf("  Total samples: %d\n\n", EPOCHS * SAMPLES_PER_EPOCH);
    
    // Training loop
    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        printf("Epoch %d/%d - ", epoch, EPOCHS);
        
        long double epoch_loss = 0.0L;
        size_t epoch_samples = 0;
        
        // Process samples for this epoch
        for (int sample = 0; sample < SAMPLES_PER_EPOCH; sample++) {
            long double prev_loss = stats.total_loss;
            train_sample(sam, text_content, text_size, &stats);
            
            if (stats.total_loss > prev_loss) {
                epoch_loss += (stats.total_loss - prev_loss);
                epoch_samples++;
            }
        }
        
        // Calculate epoch statistics
        long double avg_loss = epoch_samples > 0 ? epoch_loss / epoch_samples : 0.0L;
        time_t elapsed = time(NULL) - stats.start_time;
        
        printf("Loss: %.6Lf - Samples: %zu - Time: %lds\n", 
               avg_loss, epoch_samples, elapsed);
        
        // Test model every 5 epochs
        if (epoch % 5 == 0) {
            test_model_basic(sam);
            
            // Save checkpoint
            char checkpoint[100];
            snprintf(checkpoint, sizeof(checkpoint), "stage1_basic_epoch_%d.bin", epoch);
            if (SAM_save(sam, checkpoint) == 1) {
                printf("✓ Checkpoint saved: %s\n", checkpoint);
            }
        }
    }
    
    // Save final model
    printf("\nSaving final model...\n");
    if (SAM_save(sam, "stage1_basic_final.bin") == 1) {
        printf("✓ Final model saved: stage1_basic_final.bin\n");
    }
    
    // Final test
    printf("\n=== Final Model Test ===\n");
    test_model_basic(sam);
    
    // Training summary
    printf("\n=== Training Summary ===\n");
    printf("Total training time: %ld seconds\n", time(NULL) - stats.start_time);
    printf("Total samples processed: %zu\n", stats.samples_processed);
    if (stats.samples_processed > 0) {
        printf("Final average loss: %.6Lf\n", stats.total_loss / stats.samples_processed);
    }
    
    // Cleanup
    free(text_content);
    SAM_destroy(sam);
    
    printf("\n=== Stage 1 Basic Training Completed ===\n");
    printf("Model is ready for testing and Stage 2 training\n");
    
    return 0;
}
