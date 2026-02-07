#include "SAM/SAM.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define INPUT_DIM 256
#define OUTPUT_DIM 64
#define NUM_HEADS 8
#define DEFAULT_EPOCHS 20
#define DEFAULT_SAMPLES_PER_EPOCH 10

typedef struct {
    long double total_loss;
    size_t samples_processed;
    time_t start_time;
    int epochs;
    int samples_per_epoch;
} TrainingStats;

// Check for NaN or infinite values
int is_valid_value(long double val) {
    return !isnan(val) && !isinf(val);
}

// Check if vector contains valid values
int is_valid_vector(long double *vec, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (!is_valid_value(vec[i])) {
            return 0;
        }
    }
    return 1;
}

// Initialize model with validation
SAM_t* init_training_model() {
    printf("=== SAM AGI Stage 1: Fixed Raw Training ===\n\n");
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

// Load text content with validation
char* load_text_content(const char *filename, size_t *size) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Cannot open file %s\n", filename);
        return NULL;
    }
    
    fseek(file, 0, SEEK_END);
    *size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    if (*size < INPUT_DIM + OUTPUT_DIM) {
        printf("Error: File too small for training\n");
        fclose(file);
        return NULL;
    }
    
    char *content = malloc(*size + 1);
    if (content) {
        size_t read_size = fread(content, 1, *size, file);
        content[read_size] = '\0';
        printf("✓ Loaded %zu characters from %s\n", read_size, filename);
    }
    
    fclose(file);
    return content;
}

// Create safe training sample
int create_training_sample(const char *text, size_t text_size, 
                           long double *input, long double *target) {
    // Ensure we have enough text
    if (text_size < INPUT_DIM + OUTPUT_DIM + 10) {
        return 0; // Not enough data
    }
    
    // Pick safe starting position
    size_t max_pos = text_size - INPUT_DIM - OUTPUT_DIM - 10;
    size_t pos = rand() % max_pos;
    
    // Create input vector with validation
    for (int i = 0; i < INPUT_DIM; i++) {
        if (pos + i < text_size) {
            char c = text[pos + i];
            input[i] = (long double)c / 256.0L;
            
            // Ensure valid value
            if (!is_valid_value(input[i])) {
                input[i] = 0.0L;
            }
        } else {
            input[i] = 0.0L;
        }
    }
    
    // Create target vector with validation
    for (int i = 0; i < OUTPUT_DIM; i++) {
        if (pos + INPUT_DIM + i < text_size) {
            char c = text[pos + INPUT_DIM + i];
            target[i] = (long double)c / 256.0L;
            
            // Ensure valid value
            if (!is_valid_value(target[i])) {
                target[i] = 0.0L;
            }
        } else {
            target[i] = 0.0L;
        }
    }
    
    return 1; // Success
}

// Train one sample with safety checks
int train_sample(SAM_t *sam, const char *text, size_t text_size, TrainingStats *stats) {
    // Allocate memory
    long double *input = malloc(INPUT_DIM * sizeof(long double));
    long double *target = malloc(OUTPUT_DIM * sizeof(long double));
    
    if (!input || !target) {
        free(input);
        free(target);
        return 0;
    }
    
    // Create training sample
    if (!create_training_sample(text, text_size, input, target)) {
        free(input);
        free(target);
        return 0;
    }
    
    // Validate input and target
    if (!is_valid_vector(input, INPUT_DIM) || !is_valid_vector(target, OUTPUT_DIM)) {
        free(input);
        free(target);
        return 0;
    }
    
    // Create input sequence for SAM
    long double **input_seq = malloc(sizeof(long double*));
    input_seq[0] = input;
    
    // Train the model
    SAM_train(sam, input_seq, 1, target);
    
    // Calculate loss with validation
    long double *output = SAM_forward(sam, input_seq, 1);
    if (output && is_valid_vector(output, OUTPUT_DIM)) {
        long double sample_loss = 0.0L;
        int valid_outputs = 0;
        
        for (int i = 0; i < OUTPUT_DIM; i++) {
            if (is_valid_value(output[i]) && is_valid_value(target[i])) {
                long double error = output[i] - target[i];
                sample_loss += error * error;
                valid_outputs++;
            }
        }
        
        if (valid_outputs > 0) {
            stats->total_loss += sample_loss / valid_outputs;
            stats->samples_processed++;
        }
        
        free(output);
    }
    
    // Adapt the model
    SAM_adapt(sam, input_seq, 1);
    
    // Cleanup
    free(input_seq);
    free(input);
    free(target);
    
    return 1; // Success
}

// Safe model test
void test_model_safe(SAM_t *sam) {
    printf("\n=== Safe Model Test ===\n");
    
    // Create simple test input
    long double *test_input = calloc(INPUT_DIM, sizeof(long double));
    long double **input_seq = malloc(sizeof(long double*));
    
    if (!test_input || !input_seq) {
        printf("✗ Memory allocation failed\n");
        free(test_input);
        free(input_seq);
        return;
    }
    
    input_seq[0] = test_input;
    
    // Fill with simple pattern
    for (int i = 0; i < INPUT_DIM; i++) {
        test_input[i] = (long double)(i % 256) / 256.0L;
    }
    
    // Get output
    long double *output = SAM_forward(sam, input_seq, 1);
    if (output && is_valid_vector(output, OUTPUT_DIM)) {
        printf("✓ Forward pass successful\n");
        printf("  First 5 outputs: ");
        for (int i = 0; i < 5; i++) {
            printf("%.6Lf ", output[i]);
        }
        printf("\n");
        free(output);
    } else {
        printf("✗ Forward pass failed or invalid output\n");
    }
    
    free(input_seq);
    free(test_input);
}

int main(int argc, char *argv[]) {
    srand(time(NULL));
    
    // Parse command line arguments
    TrainingStats stats = {0};
    stats.epochs = DEFAULT_EPOCHS;
    stats.samples_per_epoch = DEFAULT_SAMPLES_PER_EPOCH;
    
    if (argc > 1) {
        stats.epochs = atoi(argv[1]);
        if (stats.epochs <= 0) stats.epochs = DEFAULT_EPOCHS;
    }
    if (argc > 2) {
        stats.samples_per_epoch = atoi(argv[2]);
        if (stats.samples_per_epoch <= 0) stats.samples_per_epoch = DEFAULT_SAMPLES_PER_EPOCH;
    }
    
    // Initialize model
    SAM_t *sam = init_training_model();
    if (!sam) return 1;
    
    // Load training data
    const char *filename = "training_data/raw_texts/Frankenstein.txt";
    if (argc > 3) filename = argv[3];
    
    printf("Loading training data: %s\n", filename);
    size_t text_size;
    char *text_content = load_text_content(filename, &text_size);
    
    if (!text_content) {
        printf("Failed to load training data\n");
        SAM_destroy(sam);
        return 1;
    }
    
    printf("\n=== Starting Safe Raw Pattern Training ===\n");
    printf("Training configuration:\n");
    printf("  Epochs: %d\n", stats.epochs);
    printf("  Samples per epoch: %d\n", stats.samples_per_epoch);
    printf("  Total samples: %d\n\n", stats.epochs * stats.samples_per_epoch);
    
    stats.start_time = time(NULL);
    
    // Training loop with safety checks
    for (int epoch = 1; epoch <= stats.epochs; epoch++) {
        printf("Epoch %d/%d - ", epoch, stats.epochs);
        
        long double epoch_loss = 0.0L;
        size_t epoch_samples = 0;
        int successful_samples = 0;
        
        // Process samples for this epoch
        for (int sample = 0; sample < stats.samples_per_epoch; sample++) {
            long double prev_loss = stats.total_loss;
            
            if (train_sample(sam, text_content, text_size, &stats)) {
                successful_samples++;
                
                if (stats.total_loss > prev_loss) {
                    epoch_loss += (stats.total_loss - prev_loss);
                    epoch_samples++;
                }
            }
        }
        
        // Calculate epoch statistics
        long double avg_loss = (epoch_samples > 0) ? epoch_loss / epoch_samples : 0.0L;
        time_t elapsed = time(NULL) - stats.start_time;
        
        printf("Loss: %.6Lf - Samples: %zu/%d - Time: %lds\n", 
               avg_loss, epoch_samples, stats.samples_per_epoch, elapsed);
        
        // Test model every 5 epochs
        if (epoch % 5 == 0) {
            test_model_safe(sam);
            
            // Save checkpoint
            char checkpoint[100];
            snprintf(checkpoint, sizeof(checkpoint), "stage1_fixed_epoch_%d.bin", epoch);
            if (SAM_save(sam, checkpoint) == 1) {
                printf("✓ Checkpoint saved: %s\n", checkpoint);
            } else {
                printf("✗ Failed to save checkpoint\n");
            }
        }
        
        // Check for numerical issues
        if (isnan(avg_loss) || isinf(avg_loss)) {
            printf("⚠️  Numerical instability detected, stopping training\n");
            break;
        }
    }
    
    // Save final model
    printf("\nSaving final model...\n");
    if (SAM_save(sam, "stage1_fixed_final.bin") == 1) {
        printf("✓ Final model saved: stage1_fixed_final.bin\n");
    } else {
        printf("✗ Failed to save final model\n");
    }
    
    // Final test
    printf("\n=== Final Model Test ===\n");
    test_model_safe(sam);
    
    // Training summary
    printf("\n=== Training Summary ===\n");
    printf("Total training time: %ld seconds\n", time(NULL) - stats.start_time);
    printf("Total samples processed: %zu\n", stats.samples_processed);
    if (stats.samples_processed > 0) {
        printf("Final average loss: %.6Lf\n", stats.total_loss / stats.samples_processed);
    } else {
        printf("No samples processed successfully\n");
    }
    
    // Cleanup
    free(text_content);
    SAM_destroy(sam);
    
    printf("\n=== Stage 1 Fixed Training Completed ===\n");
    printf("Model is ready for testing and Stage 2 training\n");
    
    return 0;
}
