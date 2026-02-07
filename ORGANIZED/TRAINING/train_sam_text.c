#include "SAM/SAM.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>

#define INPUT_DIM 256
#define OUTPUT_DIM 64
#define NUM_HEADS 8
#define MAX_LINE_LENGTH 1024
#define MAX_WORD_LENGTH 100

// Simple text preprocessing
void preprocess_text(const char *text, long double *vector, size_t vector_size) {
    // Convert text to numerical representation
    size_t len = strlen(text);
    for (size_t i = 0; i < vector_size; i++) {
        if (i < len) {
            // Simple character encoding
            vector[i] = (long double)text[i] / 256.0L;
        } else {
            vector[i] = 0.0L;
        }
    }
}

// Read CSV training data
int read_training_data(const char *filename, long double ***inputs, long double **targets, size_t *num_samples) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        return 0;
    }
    
    // Count samples first
    char line[MAX_LINE_LENGTH];
    *num_samples = 0;
    while (fgets(line, sizeof(line), file)) {
        if (strlen(line) > 1) { // Skip empty lines
            (*num_samples)++;
        }
    }
    rewind(file);
    
    // Skip header
    fgets(line, sizeof(line), file);
    (*num_samples)--;
    
    if (*num_samples == 0) {
        printf("Error: No training data found\n");
        fclose(file);
        return 0;
    }
    
    // Allocate memory
    *inputs = malloc(*num_samples * sizeof(long double*));
    *targets = malloc(*num_samples * sizeof(long double));
    
    if (!*inputs || !*targets) {
        printf("Error: Memory allocation failed\n");
        fclose(file);
        return 0;
    }
    
    // Read data
    size_t sample = 0;
    while (fgets(line, sizeof(line), file) && sample < *num_samples) {
        // Parse CSV: sequence,next_word
        char *comma = strchr(line, ',');
        if (comma) {
            *comma = '\0';
            char *sequence = line;
            char *next_word = comma + 1;
            
            // Remove newline from next_word
            char *newline = strchr(next_word, '\n');
            if (newline) *newline = '\0';
            
            // Allocate and process input
            (*inputs)[sample] = malloc(INPUT_DIM * sizeof(long double));
            if ((*inputs)[sample]) {
                preprocess_text(sequence, (*inputs)[sample], INPUT_DIM);
                
                // Process target (next word)
                (*targets)[sample] = 0.0L;
                for (size_t i = 0; i < strlen(next_word) && i < 10; i++) {
                    (*targets)[sample] += (long double)next_word[i] / 256.0L;
                }
                (*targets)[sample] /= 10.0L; // Normalize
                
                sample++;
            }
        }
    }
    
    fclose(file);
    printf("Loaded %zu training samples\n", sample);
    *num_samples = sample;
    return 1;
}

// Train SAM on text data
void train_sam_on_text(SAM_t *sam, const char *data_file, int epochs) {
    printf("Training SAM on text data from %s\n", data_file);
    
    long double **inputs = NULL;
    long double *targets = NULL;
    size_t num_samples = 0;
    
    if (!read_training_data(data_file, &inputs, &targets, &num_samples)) {
        return;
    }
    
    printf("Starting training for %d epochs...\n", epochs);
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        long double total_loss = 0.0L;
        
        for (size_t sample = 0; sample < num_samples; sample++) {
            // Create input sequence
            long double **input_seq = malloc(sizeof(long double*));
            input_seq[0] = inputs[sample];
            
            // Train SAM
            SAM_train(sam, input_seq, 1, &targets[sample]);
            
            // Calculate loss
            long double *output = SAM_forward(sam, input_seq, 1);
            if (output) {
                long double sample_loss = 0.0L;
                for (size_t i = 0; i < OUTPUT_DIM; i++) {
                    long double error = output[i] - targets[sample];
                    sample_loss += error * error;
                }
                total_loss += sample_loss / OUTPUT_DIM;
                free(output);
            }
            
            // Adapt model
            SAM_adapt(sam, input_seq, 1);
            
            free(input_seq);
        }
        
        printf("Epoch %d/%d - Avg Loss: %.6Lf\n", epoch + 1, epochs, total_loss / num_samples);
        
        // Save checkpoint every 5 epochs
        if ((epoch + 1) % 5 == 0) {
            char checkpoint_file[100];
            snprintf(checkpoint_file, sizeof(checkpoint_file), "sam_text_epoch_%d.bin", epoch + 1);
            if (SAM_save(sam, checkpoint_file) == 1) {
                printf("  ✓ Saved checkpoint to %s\n", checkpoint_file);
            }
        }
    }
    
    // Cleanup
    for (size_t i = 0; i < num_samples; i++) {
        free(inputs[i]);
    }
    free(inputs);
    free(targets);
}

// Test text generation
void test_text_generation(SAM_t *sam) {
    printf("\n=== Text Generation Test ===\n");
    
    const char *test_prompts[] = {
        "The quick brown",
        "Neural networks",
        "Machine learning",
        "Artificial intelligence"
    };
    
    for (int i = 0; i < 4; i++) {
        printf("\nPrompt: \"%s\"\n", test_prompts[i]);
        
        // Process prompt
        long double input[INPUT_DIM];
        preprocess_text(test_prompts[i], input, INPUT_DIM);
        
        // Create input sequence
        long double **input_seq = malloc(sizeof(long double*));
        input_seq[0] = input;
        
        // Generate output
        long double *output = SAM_forward(sam, input_seq, 1);
        if (output) {
            // Simple text reconstruction from output
            printf("Generated: ");
            for (int j = 0; j < 5; j++) {
                char c = (char)(output[j] * 256.0L);
                if (isalnum(c)) {
                    printf("%c", c);
                }
            }
            printf("...\n");
            
            free(output);
        }
        
        free(input_seq);
    }
}

int main(int argc, char *argv[]) {
    printf("=== SAM AGI Text Training ===\n\n");
    
    const char *data_file = "training_data.csv";
    int epochs = 20;
    
    // Parse command line arguments
    if (argc > 1) {
        data_file = argv[1];
    }
    if (argc > 2) {
        epochs = atoi(argv[2]);
    }
    
    printf("Configuration:\n");
    printf("  Data file: %s\n", data_file);
    printf("  Epochs: %d\n", epochs);
    printf("  Input dim: %d\n", INPUT_DIM);
    printf("  Output dim: %d\n", OUTPUT_DIM);
    printf("\n");
    
    // Initialize or load SAM model
    SAM_t *sam = SAM_load("sam_text_model.bin");
    if (sam) {
        printf("Loaded existing text-trained model\n");
    } else {
        printf("Initializing new SAM model for text training...\n");
        sam = SAM_init(INPUT_DIM, OUTPUT_DIM, NUM_HEADS, 0);
        if (!sam) {
            fprintf(stderr, "Failed to initialize SAM\n");
            return 1;
        }
    }
    
    // Train on text data
    train_sam_on_text(sam, data_file, epochs);
    
    // Save final model
    printf("\nSaving text-trained model...\n");
    if (SAM_save(sam, "sam_text_model.bin") == 1) {
        printf("✓ Model saved to sam_text_model.bin\n");
    }
    
    // Test text generation
    test_text_generation(sam);
    
    // Cleanup
    SAM_destroy(sam);
    
    printf("\n=== Text Training Completed ===\n");
    return 0;
}
