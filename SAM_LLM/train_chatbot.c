#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>
#include "../SAM/SAM.h"
#include "../utils/NN/transformer.h"

#define MAX_WORD_LENGTH 100
#define SEQUENCE_LENGTH 10
#define MODEL_DIM 256
#define OUTPUT_DIM 256
#define NUM_HEADS 8
#define MAX_LINE_LENGTH 8192

// Structure to hold training data
typedef struct {
    char*** sequences;  // Array of word sequences
    char** targets;     // Array of target words
    size_t num_samples; // Number of training samples
    size_t seq_length;  // Length of each sequence
} TrainingData;

// Function to check if file is a text file
int is_text_file(const char* filename) {
    const char* ext = strrchr(filename, '.');
    if (!ext) return 0;
    return (strcmp(ext, ".txt") == 0 || strcmp(ext, ".c") == 0 || 
            strcmp(ext, ".h") == 0 || strcmp(ext, ".md") == 0);
}

// Function to extract words from text
size_t extract_words(const char* text, char** words, size_t max_words) {
    size_t word_count = 0;
    char* text_copy = strdup(text);
    char* word = strtok(text_copy, " \t\n\r.,!?;:()[]{}\"'-");
    
    while (word && word_count < max_words) {
        // Skip empty strings
        if (strlen(word) > 0) {
            words[word_count] = strdup(word);
            word_count++;
        }
        word = strtok(NULL, " \t\n\r.,!?;:()[]{}\"'-");
    }
    
    free(text_copy);
    return word_count;
}

// Function to load all text files from directory
TrainingData* load_all_text_files(const char* dir_path, size_t sequence_length) {
    TrainingData* data = (TrainingData*)malloc(sizeof(TrainingData));
    data->seq_length = sequence_length;
    data->num_samples = 0;
    data->sequences = NULL;
    data->targets = NULL;
    
    // First pass: count total samples
    DIR* dir = opendir(dir_path);
    if (!dir) {
        fprintf(stderr, "Error opening directory: %s\n", dir_path);
        free(data);
        return NULL;
    }
    
    size_t capacity = 10000;
    data->sequences = (char***)malloc(capacity * sizeof(char**));
    data->targets = (char**)malloc(capacity * sizeof(char*));
    
    struct dirent* entry;
    char** words = (char**)malloc(10000 * sizeof(char*));
    
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue;
        
        char filepath[1024];
        snprintf(filepath, sizeof(filepath), "%s/%s", dir_path, entry->d_name);
        
        struct stat st;
        if (stat(filepath, &st) != 0) continue;
        if (!S_ISREG(st.st_mode)) continue;
        if (!is_text_file(entry->d_name)) continue;
        
        FILE* f = fopen(filepath, "r");
        if (!f) continue;
        
        char line[MAX_LINE_LENGTH];
        while (fgets(line, sizeof(line), f)) {
            // Extract words from line
            size_t word_count = extract_words(line, words, 10000);
            
            if (word_count < sequence_length + 1) continue;
            
            // Create sequences
            for (size_t i = 0; i <= word_count - sequence_length - 1; i++) {
                if (data->num_samples >= capacity) {
                    capacity *= 2;
                    data->sequences = (char***)realloc(data->sequences, capacity * sizeof(char**));
                    data->targets = (char**)realloc(data->targets, capacity * sizeof(char*));
                }
                
                // Allocate sequence
                data->sequences[data->num_samples] = (char**)malloc(sequence_length * sizeof(char*));
                for (size_t j = 0; j < sequence_length; j++) {
                    data->sequences[data->num_samples][j] = strdup(words[i + j]);
                }
                
                // Set target
                data->targets[data->num_samples] = strdup(words[i + sequence_length]);
                
                data->num_samples++;
            }
        }
        
        fclose(f);
    }
    
    closedir(dir);
    
    // Free temporary words array
    for (size_t i = 0; i < 10000; i++) {
        if (words[i]) free(words[i]);
    }
    free(words);
    
    printf("Loaded %zu training samples from %s\n", data->num_samples, dir_path);
    return data;
}

// Function to free training data
void free_training_data(TrainingData* data) {
    if (!data) return;
    
    for (size_t i = 0; i < data->num_samples; i++) {
        for (size_t j = 0; j < data->seq_length; j++) {
            free(data->sequences[i][j]);
        }
        free(data->sequences[i]);
        free(data->targets[i]);
    }
    free(data->sequences);
    free(data->targets);
    free(data);
}

// Function to encode word into vector
void encode_word(const char* word, long double* encoded, size_t dim) {
    memset(encoded, 0, dim * sizeof(long double));
    size_t len = strlen(word);
    size_t copy_len = (len < dim) ? len : dim;
    
    for (size_t i = 0; i < copy_len; i++) {
        encoded[i] = ((long double)((unsigned char)word[i])) / 255.0L;
    }
}

// Function to encode sequence of words
void encode_sequence(char** words, size_t num_words, long double* encoded, size_t dim) {
    memset(encoded, 0, dim * sizeof(long double));
    
    if (num_words == 0) return;
    
    // Average all word vectors
    for (size_t w = 0; w < num_words; w++) {
        if (!words[w]) continue;
        
        size_t len = strlen(words[w]);
        size_t copy_len = (len < dim) ? len : dim;
        
        for (size_t i = 0; i < copy_len; i++) {
            encoded[i] += ((long double)((unsigned char)words[w][i])) / 255.0L;
        }
    }
    
    // Average
    for (size_t i = 0; i < dim; i++) {
        encoded[i] /= num_words;
    }
}

int main(int argc, char* argv[]) {
    // Default to DATASETS directory, prioritizing RomeoAndJuliet and Frankenstein
    // Arguments: [directory] [epochs]
    const char* datasets_dir = "../utils/DATASETS";
    const char* tests_dir = "../utils/TESTS";
    size_t num_epochs = 10;
    
    // Parse arguments
    if (argc > 1) {
        // Check if first arg is a number (epochs) or directory
        if (atoi(argv[1]) > 0 && strlen(argv[1]) < 10) {
            // It's a number, treat as epochs
            num_epochs = atoi(argv[1]);
        } else {
            // It's a directory
            datasets_dir = argv[1];
            if (argc > 2) {
                num_epochs = atoi(argv[2]);
            }
        }
    }
    if (argc > 2 && atoi(argv[2]) > 0) {
        num_epochs = atoi(argv[2]);
    }
    
    const long double learning_rate = 0.001L;
    
    printf("=== SAM Chatbot Training ===\n\n");
    printf("Loading training data from:\n");
    printf("  - Primary: %s (Romeo & Juliet, Frankenstein)\n", datasets_dir);
    printf("  - Secondary: %s (test files)\n\n", tests_dir);
    
    // Load all text files from DATASETS directory first (prioritize stories)
    TrainingData* train_data = load_all_text_files(datasets_dir, SEQUENCE_LENGTH);
    size_t datasets_samples = train_data ? train_data->num_samples : 0;
    
    // Also load from TESTS directory and merge
    TrainingData* test_data = load_all_text_files(tests_dir, SEQUENCE_LENGTH);
    size_t tests_samples = test_data ? test_data->num_samples : 0;
    
    if (test_data && test_data->num_samples > 0) {
        // Merge test_data into train_data
        size_t old_count = train_data ? train_data->num_samples : 0;
        size_t new_count = old_count + test_data->num_samples;
        
        if (train_data) {
            // Reallocate to fit both
            train_data->sequences = (char***)realloc(train_data->sequences, new_count * sizeof(char**));
            train_data->targets = (char**)realloc(train_data->targets, new_count * sizeof(char*));
        } else {
            // Create new if train_data was NULL
            train_data = (TrainingData*)malloc(sizeof(TrainingData));
            train_data->seq_length = SEQUENCE_LENGTH;
            train_data->sequences = (char***)malloc(new_count * sizeof(char**));
            train_data->targets = (char**)malloc(new_count * sizeof(char*));
            train_data->num_samples = 0;
        }
        
        // Copy test_data samples
        for (size_t i = 0; i < test_data->num_samples; i++) {
            train_data->sequences[old_count + i] = test_data->sequences[i];
            train_data->targets[old_count + i] = test_data->targets[i];
        }
        
        train_data->num_samples = new_count;
        
        // Free test_data structure (but not the actual data, it's now in train_data)
        free(test_data->sequences);
        free(test_data->targets);
        free(test_data);
    }
    
    if (!train_data || train_data->num_samples == 0) {
        fprintf(stderr, "Failed to load training data or no samples found\n");
        return 1;
    }
    
    printf("Loaded %zu training samples total\n", train_data->num_samples);
    printf("  - From DATASETS (%s): %zu samples\n", datasets_dir, datasets_samples);
    if (tests_samples > 0) {
        printf("  - From TESTS (%s): %zu samples\n", tests_dir, tests_samples);
    }
    printf("\n");
    
    // Initialize SAM model
    printf("Initializing SAM model...\n");
    SAM_t* sam = SAM_init(MODEL_DIM, OUTPUT_DIM, NUM_HEADS, 0);
    if (!sam) {
        fprintf(stderr, "Failed to initialize SAM model\n");
        free_training_data(train_data);
        return 1;
    }
    printf("SAM model initialized\n\n");
    
    // Training loop
    printf("Starting training for %zu epochs...\n\n", num_epochs);
    
    for (size_t epoch = 0; epoch < num_epochs; epoch++) {
        long double epoch_loss = 0.0L;
        size_t samples_processed = 0;
        
        for (size_t i = 0; i < train_data->num_samples; i++) {
            // Encode input sequence
            long double* input_vec = (long double*)calloc(MODEL_DIM, sizeof(long double));
            encode_sequence(train_data->sequences[i], SEQUENCE_LENGTH, input_vec, MODEL_DIM);
            
            // Encode target
            long double* target = (long double*)calloc(OUTPUT_DIM, sizeof(long double));
            encode_word(train_data->targets[i], target, OUTPUT_DIM);
            
            // Create input sequence for SAM
            long double** input_seq = (long double**)malloc(sizeof(long double*));
            input_seq[0] = input_vec;
            
            // Forward pass
            long double* output = SAM_forward(sam, input_seq, 1);
            
            if (output) {
                // Calculate loss (MSE)
                long double sample_loss = 0.0L;
                for (size_t j = 0; j < OUTPUT_DIM; j++) {
                    long double diff = output[j] - target[j];
                    sample_loss += diff * diff;
                }
                sample_loss /= OUTPUT_DIM;
                epoch_loss += sample_loss;
                samples_processed++;
                
                // Train the model
                SAM_train(sam, input_seq, 1, target);
                
                free(output);
            }
            
            free(input_vec);
            free(target);
            free(input_seq);
            
            // Print progress
            if ((i + 1) % 50 == 0 || (i + 1) == train_data->num_samples) {
                printf("\rEpoch %zu, Sample %zu/%zu, Avg Loss: %.6Lf", 
                       epoch + 1, i + 1, train_data->num_samples, 
                       samples_processed > 0 ? epoch_loss / samples_processed : 0.0L);
                fflush(stdout);
            }
        }
        
        printf("\nEpoch %zu completed. Average loss: %.6Lf\n\n", 
               epoch + 1, epoch_loss / samples_processed);
    }
    
    // Save model
    printf("Saving trained model...\n");
    char timestamp[64];
    time_t rawtime;
    struct tm *info;
    time(&rawtime);
    info = localtime(&rawtime);
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", info);
    
    char filename_timestamped[128];
    snprintf(filename_timestamped, sizeof(filename_timestamped), "../sam_chatbot_%s.bin", timestamp);
    
    if (SAM_save(sam, filename_timestamped) == 1) {
        printf("✓ Model saved to %s\n", filename_timestamped);
    } else {
        printf("✗ Failed to save model to %s\n", filename_timestamped);
    }
    
    if (SAM_save(sam, "../sam_trained_model.bin") == 1) {
        printf("✓ Model saved to ../sam_trained_model.bin (default)\n");
    } else {
        printf("✗ Failed to save model to ../sam_trained_model.bin\n");
    }
    
    // Cleanup
    SAM_destroy(sam);
    free_training_data(train_data);
    
    printf("\n=== Training Complete ===\n");
    return 0;
}

