#include "SAM/SAM.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>

#define INPUT_DIM 512
#define OUTPUT_DIM 128
#define NUM_HEADS 12
#define STAGE1_EPOCHS 50
#define BATCH_SIZE 16
#define LEARNING_RATE 0.01L
#define SEQUENCE_LENGTH 128
#define MAX_FILE_SIZE 1000000 // 1MB per file max

// Training statistics
typedef struct {
    long double total_loss;
    long double avg_loss;
    size_t samples_processed;
    time_t start_time;
    time_t current_time;
} TrainingStats;

// File processing
typedef struct {
    char filename[256];
    size_t size;
    char *content;
} TextFile;

// Initialize expanded SAM model for Stage 1
SAM_t* init_stage1_model() {
    printf("Initializing Stage 1 SAM Model (Raw Pattern Learning)...\n");
    printf("Configuration:\n");
    printf("  Input Dimension: %d\n", INPUT_DIM);
    printf("  Output Dimension: %d\n", OUTPUT_DIM);
    printf("  Attention Heads: %d\n", NUM_HEADS);
    printf("  Sequence Length: %d\n", SEQUENCE_LENGTH);
    printf("\n");
    
    SAM_t *sam = SAM_init(INPUT_DIM, OUTPUT_DIM, NUM_HEADS, 0);
    if (!sam) {
        fprintf(stderr, "Failed to initialize SAM model\n");
        return NULL;
    }
    
    printf("✓ Stage 1 model initialized successfully\n");
    printf("  - Submodels: %zu\n", sam->num_submodels);
    printf("  - Context: %.2Lf\n", sam->context);
    printf("\n");
    
    return sam;
}

// Load text files from directory
int load_text_files(const char *directory, TextFile **files, size_t *count) {
    DIR *dir;
    struct dirent *entry;
    struct stat file_stat;
    
    dir = opendir(directory);
    if (!dir) {
        printf("Error: Cannot open directory %s\n", directory);
        return 0;
    }
    
    // Count files first
    *count = 0;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) { // Regular files only
            char filepath[512];
            snprintf(filepath, sizeof(filepath), "%s/%s", directory, entry->d_name);
            
            if (stat(filepath, &file_stat) == 0 && file_stat.st_size < MAX_FILE_SIZE) {
                (*count)++;
            }
        }
    }
    rewinddir(dir);
    
    if (*count == 0) {
        printf("No suitable text files found in %s\n", directory);
        closedir(dir);
        return 0;
    }
    
    // Allocate memory for files
    *files = malloc(*count * sizeof(TextFile));
    if (!*files) {
        printf("Memory allocation failed\n");
        closedir(dir);
        return 0;
    }
    
    // Load files
    size_t index = 0;
    while ((entry = readdir(dir)) != NULL && index < *count) {
        if (entry->d_type == DT_REG) {
            char filepath[512];
            snprintf(filepath, sizeof(filepath), "%s/%s", directory, entry->d_name);
            
            if (stat(filepath, &file_stat) == 0 && file_stat.st_size < MAX_FILE_SIZE) {
                FILE *file = fopen(filepath, "r");
                if (file) {
                    TextFile *tf = &(*files)[index];
                    strncpy(tf->filename, entry->d_name, 255);
                    tf->filename[255] = '\0';
                    tf->size = file_stat.st_size;
                    
                    tf->content = malloc(tf->size + 1);
                    if (tf->content) {
                        size_t read_size = fread(tf->content, 1, tf->size, file);
                        tf->content[read_size] = '\0';
                        
                        printf("Loaded: %s (%zu bytes)\n", tf->filename, tf->size);
                        index++;
                    }
                    fclose(file);
                }
            }
        }
    }
    
    closedir(dir);
    *count = index;
    printf("✓ Loaded %zu text files\n", *count);
    return 1;
}

// Convert text to numerical representation
void text_to_vectors(const char *text, long double **inputs, long double *targets, 
                     size_t sequence_length, size_t input_dim, size_t output_dim) {
    size_t text_len = strlen(text);
    
    // Create input sequences (sliding window)
    for (size_t i = 0; i < sequence_length; i++) {
        for (size_t j = 0; j < input_dim; j++) {
            if (i * input_dim + j < text_len) {
                // Character encoding with position
                char c = text[i * input_dim + j];
                inputs[i][j] = (long double)c / 256.0L + (long double)i * 0.001L;
            } else {
                inputs[i][j] = 0.0L;
            }
        }
    }
    
    // Create target (next character prediction)
    for (size_t i = 0; i < output_dim; i++) {
        if (sequence_length * input_dim + i < text_len) {
            char c = text[sequence_length * input_dim + i];
            targets[i] = (long double)c / 256.0L;
        } else {
            targets[i] = 0.0L;
        }
    }
}

// Training batch processing
void process_training_batch(SAM_t *sam, TextFile *files, size_t file_count, 
                           TrainingStats *stats, int epoch) {
    size_t samples_per_file = BATCH_SIZE / file_count;
    if (samples_per_file == 0) samples_per_file = 1;
    
    for (size_t file_idx = 0; file_idx < file_count; file_idx++) {
        TextFile *file = &files[file_idx];
        
        for (size_t sample = 0; sample < samples_per_file; sample++) {
            // Random position in file
            size_t start_pos = rand() % (file->size - SEQUENCE_LENGTH * INPUT_DIM - OUTPUT_DIM);
            const char *text_segment = &file->content[start_pos];
            
            // Allocate memory for this sample
            long double **inputs = malloc(SEQUENCE_LENGTH * sizeof(long double*));
            long double *targets = malloc(OUTPUT_DIM * sizeof(long double));
            
            if (!inputs || !targets) {
                free(inputs);
                free(targets);
                continue;
            }
            
            for (size_t i = 0; i < SEQUENCE_LENGTH; i++) {
                inputs[i] = malloc(INPUT_DIM * sizeof(long double));
                if (!inputs[i]) {
                    for (size_t j = 0; j < i; j++) free(inputs[j]);
                    free(inputs);
                    free(targets);
                    continue;
                }
            }
            
            // Convert text to vectors
            text_to_vectors(text_segment, inputs, targets, SEQUENCE_LENGTH, INPUT_DIM, OUTPUT_DIM);
            
            // Train SAM
            SAM_train(sam, inputs, SEQUENCE_LENGTH, targets);
            
            // Calculate loss
            long double *output = SAM_forward(sam, inputs, SEQUENCE_LENGTH);
            if (output) {
                long double batch_loss = 0.0L;
                for (size_t i = 0; i < OUTPUT_DIM; i++) {
                    long double error = output[i] - targets[i];
                    batch_loss += error * error;
                }
                stats->total_loss += batch_loss / OUTPUT_DIM;
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
    }
}

// Save training checkpoint
void save_checkpoint(SAM_t *sam, int epoch, TrainingStats *stats) {
    char filename[256];
    snprintf(filename, sizeof(filename), "stage1_checkpoint_epoch_%d.bin", epoch);
    
    if (SAM_save(sam, filename) == 1) {
        printf("  ✓ Checkpoint saved: %s\n", filename);
        
        // Save training stats
        char stats_file[256];
        snprintf(stats_file, sizeof(stats_file), "stage1_stats_epoch_%d.txt", epoch);
        FILE *fp = fopen(stats_file, "w");
        if (fp) {
            fprintf(fp, "Epoch: %d\n", epoch);
            fprintf(fp, "Total Loss: %.6Lf\n", stats->total_loss);
            fprintf(fp, "Average Loss: %.6Lf\n", stats->avg_loss);
            fprintf(fp, "Samples Processed: %zu\n", stats->samples_processed);
            fprintf(fp, "Training Time: %ld seconds\n", stats->current_time - stats->start_time);
            fclose(fp);
        }
    } else {
        printf("  ✗ Failed to save checkpoint\n");
    }
}

// Generate sample text for evaluation
void generate_sample_text(SAM_t *sam) {
    printf("\n=== Sample Generation ===\n");
    
    // Test prompts
    const char *prompts[] = {
        "The future of artificial intelligence",
        "In the beginning there was",
        "Machine learning models can",
        "The universe is composed of"
    };
    
    for (int i = 0; i < 4; i++) {
        printf("Prompt: \"%s\"\n", prompts[i]);
        
        // Convert prompt to input
        long double **inputs = malloc(SEQUENCE_LENGTH * sizeof(long double*));
        for (size_t j = 0; j < SEQUENCE_LENGTH; j++) {
            inputs[j] = malloc(INPUT_DIM * sizeof(long double));
            text_to_vectors(prompts[i], &inputs[j], NULL, 1, INPUT_DIM, OUTPUT_DIM);
        }
        
        // Generate output
        long double *targets = malloc(OUTPUT_DIM * sizeof(long double));
        long double *output = SAM_forward(sam, inputs, SEQUENCE_LENGTH);
        
        if (output) {
            printf("Generated: ");
            for (int j = 0; j < 10; j++) {
                char c = (char)(output[j] * 256.0L);
                if (c >= 32 && c <= 126) { // Printable characters
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
        free(targets);
    }
}

// Main Stage 1 training
int main(int argc, char *argv[]) {
    printf("=== SAM AGI Stage 1: Raw Pattern Learning ===\n\n");
    
    const char *data_directory = "utils/DATASETS";
    if (argc > 1) {
        data_directory = argv[1];
    }
    
    // Initialize model
    SAM_t *sam = init_stage1_model();
    if (!sam) return 1;
    
    // Load training data
    TextFile *files;
    size_t file_count;
    
    printf("Loading training data from: %s\n", data_directory);
    if (!load_text_files(data_directory, &files, &file_count)) {
        printf("Failed to load training data\n");
        SAM_destroy(sam);
        return 1;
    }
    
    // Initialize training statistics
    TrainingStats stats = {0};
    stats.start_time = time(NULL);
    
    printf("\n=== Starting Stage 1 Training ===\n");
    printf("Configuration:\n");
    printf("  Epochs: %d\n", STAGE1_EPOCHS);
    printf("  Batch Size: %d\n", BATCH_SIZE);
    printf("  Learning Rate: %.3Lf\n", LEARNING_RATE);
    printf("  Files: %zu\n", file_count);
    printf("\n");
    
    // Training loop
    for (int epoch = 1; epoch <= STAGE1_EPOCHS; epoch++) {
        printf("Epoch %d/%d - ", epoch, STAGE1_EPOCHS);
        
        // Reset epoch stats
        long double epoch_loss = 0.0L;
        size_t epoch_samples = 0;
        
        // Process training batches
        for (int batch = 0; batch < 10; batch++) { // 10 batches per epoch
            process_training_batch(sam, files, file_count, &stats, epoch);
            epoch_loss += stats.total_loss;
            epoch_samples += stats.samples_processed;
        }
        
        // Calculate epoch statistics
        stats.current_time = time(NULL);
        stats.avg_loss = epoch_loss / epoch_samples;
        
        printf("Loss: %.6Lf - Samples: %zu - Time: %lds\n", 
               stats.avg_loss, epoch_samples, stats.current_time - stats.start_time);
        
        // Generate sample text every 5 epochs
        if (epoch % 5 == 0) {
            generate_sample_text(sam);
        }
        
        // Save checkpoint every 5 epochs
        if (epoch % 5 == 0) {
            save_checkpoint(sam, epoch, &stats);
        }
        
        // Reset for next epoch
        stats.total_loss = 0.0L;
        stats.samples_processed = 0;
    }
    
    // Save final model
    printf("\nSaving final Stage 1 model...\n");
    if (SAM_save(sam, "stage1_final_model.bin") == 1) {
        printf("✓ Stage 1 model saved successfully\n");
    }
    
    // Final evaluation
    printf("\n=== Final Evaluation ===\n");
    generate_sample_text(sam);
    
    // Cleanup
    for (size_t i = 0; i < file_count; i++) {
        free(files[i].content);
    }
    free(files);
    SAM_destroy(sam);
    
    printf("\n=== Stage 1 Training Completed ===\n");
    printf("Total training time: %ld seconds\n", time(NULL) - stats.start_time);
    printf("Ready for Stage 2: Coherence Development\n");
    
    return 0;
}
