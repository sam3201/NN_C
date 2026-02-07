#include "SAM/SAM.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Use proven working dimensions
#define INPUT_DIM 256
#define OUTPUT_DIM 64
#define NUM_HEADS 8
#define STAGE1_EPOCHS 20
#define BATCH_SIZE 4
#define LEARNING_RATE 0.01L
#define SEQUENCE_LENGTH 32

typedef struct {
    long double total_loss;
    size_t samples_processed;
    time_t start_time;
} TrainingStats;

// Initialize conservative Stage 1 model
SAM_t* init_conservative_model() {
    printf("Initializing Conservative Stage 1 SAM Model...\n");
    printf("Configuration:\n");
    printf("  Input Dimension: %d\n", INPUT_DIM);
    printf("  Output Dimension: %d\n", OUTPUT_DIM);
    printf("  Attention Heads: %d\n", NUM_HEADS);
    printf("  Sequence Length: %d\n", SEQUENCE_LENGTH);
    printf("  Epochs: %d\n", STAGE1_EPOCHS);
    printf("\n");
    
    SAM_t *sam = SAM_init(INPUT_DIM, OUTPUT_DIM, NUM_HEADS, 0);
    if (!sam) {
        fprintf(stderr, "Failed to initialize SAM model\n");
        return NULL;
    }
    
    printf("✓ Conservative model initialized successfully\n");
    printf("  - Submodels: %zu\n", sam->num_submodels);
    printf("\n");
    
    return sam;
}

// Load text content from file
char* load_text_file(const char *filename, size_t *size) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Cannot open file %s\n", filename);
        return NULL;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    *size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Allocate and read content
    char *content = malloc(*size + 1);
    if (content) {
        size_t read_size = fread(content, 1, *size, file);
        content[read_size] = '\0';
    }
    
    fclose(file);
    return content;
}

// Convert text to numerical vectors
void text_to_vectors(const char *text, long double **inputs, long double *targets) {
    size_t text_len = strlen(text);
    
    // Create input sequences
    for (size_t i = 0; i < SEQUENCE_LENGTH; i++) {
        for (size_t j = 0; j < INPUT_DIM; j++) {
            size_t pos = i * (INPUT_DIM / SEQUENCE_LENGTH) + j;
            if (pos < text_len) {
                char c = text[pos];
                inputs[i][j] = (long double)c / 256.0L;
            } else {
                inputs[i][j] = 0.0L;
            }
        }
    }
    
    // Create target (next character prediction)
    for (size_t i = 0; i < OUTPUT_DIM; i++) {
        size_t pos = SEQUENCE_LENGTH * (INPUT_DIM / SEQUENCE_LENGTH) + i;
        if (pos < text_len) {
            char c = text[pos];
            targets[i] = (long double)c / 256.0L;
        } else {
            targets[i] = 0.0L;
        }
    }
}

// Process one training sample
void process_sample(SAM_t *sam, const char *text, TrainingStats *stats) {
    // Allocate memory for inputs
    long double **inputs = malloc(SEQUENCE_LENGTH * sizeof(long double*));
    long double *targets = malloc(OUTPUT_DIM * sizeof(long double));
    
    if (!inputs || !targets) {
        free(inputs);
        free(targets);
        return;
    }
    
    // Allocate input sequences
    for (size_t i = 0; i < SEQUENCE_LENGTH; i++) {
        inputs[i] = malloc(INPUT_DIM * sizeof(long double));
        if (!inputs[i]) {
            for (size_t j = 0; j < i; j++) free(inputs[j]);
            free(inputs);
            free(targets);
            return;
        }
    }
    
    // Convert text to vectors
    text_to_vectors(text, inputs, targets);
    
    // Train SAM
    SAM_train(sam, inputs, SEQUENCE_LENGTH, targets);
    
    // Calculate loss
    long double *output = SAM_forward(sam, inputs, SEQUENCE_LENGTH);
    if (output) {
        long double sample_loss = 0.0L;
        for (size_t i = 0; i < OUTPUT_DIM; i++) {
            long double error = output[i] - targets[i];
            sample_loss += error * error;
        }
        stats->total_loss += sample_loss / OUTPUT_DIM;
        stats->samples_processed++;
        free(output);
    }
    
    // Adapt model
    SAM_adapt(sam, inputs, SEQUENCE_LENGTH);
    
    // Cleanup
    for (size_t i = 0; i < SEQUENCE_LENGTH; i++) {
        free(inputs[i]);
    }
    free(inputs);
    free(targets);
}

// Generate sample text
void generate_sample_text(SAM_t *sam) {
    printf("\n=== Sample Generation ===\n");
    
    const char *prompts[] = {
        "The future of AI",
        "Machine learning",
        "Neural networks",
        "Deep learning"
    };
    
    for (int i = 0; i < 4; i++) {
        printf("Prompt: \"%s\"\n", prompts[i]);
        
        // Create input sequence
        long double **inputs = malloc(SEQUENCE_LENGTH * sizeof(long double*));
        for (size_t j = 0; j < SEQUENCE_LENGTH; j++) {
            inputs[j] = malloc(INPUT_DIM * sizeof(long double));
        }
        
        text_to_vectors(prompts[i], inputs, NULL);
        
        // Generate output
        long double *output = SAM_forward(sam, inputs, SEQUENCE_LENGTH);
        if (output) {
            printf("Generated: ");
            for (int j = 0; j < 8; j++) {
                char c = (char)(output[j] * 256.0L);
                if (c >= 32 && c <= 126) {
                    printf("%c", c);
                }
            }
            printf("...\n");
            free(output);
        }
        
        // Cleanup
        for (size_t j = 0; j < SEQUENCE_LENGTH; j++) {
            free(inputs[j]);
        }
        free(inputs);
    }
}

int main(int argc, char *argv[]) {
    printf("=== SAM AGI Stage 1: Conservative Training ===\n\n");
    
    // Initialize model
    SAM_t *sam = init_conservative_model();
    if (!sam) return 1;
    
    // Load training data
    const char *data_file = "training_data/raw_texts/Frankenstein.txt";
    if (argc > 1) {
        data_file = argv[1];
    }
    
    printf("Loading training data: %s\n", data_file);
    size_t text_size;
    char *text_content = load_text_file(data_file, &text_size);
    
    if (!text_content) {
        printf("Failed to load training data\n");
        SAM_destroy(sam);
        return 1;
    }
    
    printf("✓ Loaded %zu characters\n\n", text_size);
    
    // Initialize training statistics
    TrainingStats stats = {0};
    stats.start_time = time(NULL);
    
    printf("=== Starting Conservative Stage 1 Training ===\n");
    
    // Training loop
    for (int epoch = 1; epoch <= STAGE1_EPOCHS; epoch++) {
        printf("Epoch %d/%d - ", epoch, STAGE1_EPOCHS);
        
        // Process multiple samples per epoch
        long double epoch_loss = 0.0L;
        size_t epoch_samples = 0;
        
        for (int batch = 0; batch < 10; batch++) {
            // Random position in text
            size_t max_pos = text_size - SEQUENCE_LENGTH * (INPUT_DIM / SEQUENCE_LENGTH) - OUTPUT_DIM;
            if (max_pos > 100) {
                size_t pos = rand() % max_pos;
                const char *text_segment = &text_content[pos];
                
                // Process sample
                long double prev_loss = stats.total_loss;
                process_sample(sam, text_segment, &stats);
                
                if (stats.total_loss > prev_loss) {
                    epoch_loss += (stats.total_loss - prev_loss);
                    epoch_samples++;
                }
            }
        }
        
        // Calculate epoch statistics
        long double avg_loss = epoch_samples > 0 ? epoch_loss / epoch_samples : 0.0L;
        time_t current_time = time(NULL);
        
        printf("Loss: %.6Lf - Samples: %zu - Time: %lds\n", 
               avg_loss, epoch_samples, current_time - stats.start_time);
        
        // Generate sample text every 5 epochs
        if (epoch % 5 == 0) {
            generate_sample_text(sam);
            
            // Save checkpoint
            char checkpoint_file[100];
            snprintf(checkpoint_file, sizeof(checkpoint_file), "stage1_conservative_epoch_%d.bin", epoch);
            if (SAM_save(sam, checkpoint_file) == 1) {
                printf("✓ Checkpoint saved: %s\n", checkpoint_file);
            }
        }
        
        // Reset epoch stats
        epoch_loss = 0.0L;
        epoch_samples = 0;
    }
    
    // Save final model
    printf("\nSaving final conservative model...\n");
    if (SAM_save(sam, "stage1_conservative_final.bin") == 1) {
        printf("✓ Conservative Stage 1 model saved\n");
    }
    
    // Final evaluation
    printf("\n=== Final Evaluation ===\n");
    generate_sample_text(sam);
    
    // Cleanup
    free(text_content);
    SAM_destroy(sam);
    
    printf("\n=== Conservative Stage 1 Training Completed ===\n");
    printf("Total training time: %ld seconds\n", time(NULL) - stats.start_time);
    printf("Total samples processed: %zu\n", stats.samples_processed);
    printf("Final average loss: %.6Lf\n", stats.total_loss / stats.samples_processed);
    
    return 0;
}
