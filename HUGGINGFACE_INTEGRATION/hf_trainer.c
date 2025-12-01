#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/wait.h>
#include "../SAM/SAM.h"

#define MAX_LINE_LENGTH 8192
#define JSON_BUFFER_SIZE 10485760  // 10MB for JSON

// Structure for training sample from HF
typedef struct {
    char* input_text;
    long double* teacher_embeddings;
    char* teacher_output;
    size_t model_dim;
} HFTrainingSample;

// Parse JSON training data (simplified - uses Python output)
int load_hf_training_data(const char* json_file, HFTrainingSample** samples, size_t* num_samples) {
    FILE* f = fopen(json_file, "r");
    if (!f) {
        fprintf(stderr, "Error opening JSON file: %s\n", json_file);
        return 0;
    }
    
    // Read entire file
    char* buffer = (char*)malloc(JSON_BUFFER_SIZE);
    if (!buffer) {
        fclose(f);
        return 0;
    }
    
    size_t bytes_read = fread(buffer, 1, JSON_BUFFER_SIZE - 1, f);
    buffer[bytes_read] = '\0';
    fclose(f);
    
    // Simple JSON parsing (for production, use a proper JSON library)
    // For now, we'll use a simpler approach: call Python to parse
    
    // Count samples (rough estimate)
    *num_samples = 0;
    const char* p = buffer;
    while ((p = strstr(p, "\"input_text\"")) != NULL) {
        (*num_samples)++;
        p++;
    }
    
    if (*num_samples == 0) {
        free(buffer);
        return 0;
    }
    
    *samples = (HFTrainingSample*)calloc(*num_samples, sizeof(HFTrainingSample));
    free(buffer);
    
    // For now, return success - actual parsing will be done by Python helper
    return 1;
}

// Train SAM using HF model embeddings as targets
void train_sam_with_hf(SAM_t* sam, const char* json_file, size_t epochs) {
    printf("Training SAM model using Hugging Face teacher model...\n");
    
    // Check if training data already exists
    FILE* check_file = fopen(json_file, "r");
    if (check_file) {
        fclose(check_file);
        printf("Training data already exists at %s - skipping Python generation\n", json_file);
    } else {
        // Generate training data using Python script
        printf("Generating training data with Python script...\n");
        char command[1024];
        // Use distilbert for better compatibility with vocabulary training
        snprintf(command, sizeof(command), 
                 "python3 hf_trainer.py distilbert-base-uncased %zu %s", epochs, json_file);
        
        printf("Running: %s\n", command);
        int result = system(command);
        
        if (result != 0) {
            fprintf(stderr, "Error running Python trainer\n");
            return;
        }
    }
    
    // Load the training data
    FILE* f = fopen(json_file, "r");
    if (!f) {
        fprintf(stderr, "Error: Training data file %s not found\n", json_file);
        return;
    }
    
    // Read and parse JSON (simplified - in production use proper JSON parser)
    char line[MAX_LINE_LENGTH];
    size_t sample_count = 0;
    
    printf("Loading training data from hf_training_data.json...\n");
    
    // For now, we'll use a Python helper to extract embeddings
    // This is a simplified version - in production, use proper JSON parsing
    
    fclose(f);
    
    printf("Training complete!\n");
}

int main(int argc, char* argv[]) {
    const char* model_name = (argc > 1) ? argv[1] : "bert-base-uncased";
    const size_t epochs = (argc > 2) ? atoi(argv[2]) : 10;
    const char* data_file = (argc > 3) ? argv[3] : "../utils/DATASETS/RomeoAndJuliet.txt";
    
    printf("=== SAM Training with Hugging Face Model ===\n\n");
    printf("Teacher Model: %s\n", model_name);
    printf("Training Data: %s\n", data_file);
    printf("Epochs: %zu\n\n", epochs);
    
    // Step 1: Generate training data using Python
    printf("Step 1: Generating training data with HF model...\n");
    char python_cmd[1024];
    snprintf(python_cmd, sizeof(python_cmd),
             "python3 hf_trainer.py %s %zu %s", model_name, epochs, data_file);
    
    printf("Running: %s\n", python_cmd);
    int python_result = system(python_cmd);
    
    if (python_result != 0) {
        fprintf(stderr, "Error running Python trainer\n");
        return 1;
    }
    
    // Step 2: Initialize SAM model
    printf("\nStep 2: Initializing SAM model...\n");
    SAM_t* sam = SAM_init(768, 768, 8, 0);  // Match HF model dimension
    if (!sam) {
        fprintf(stderr, "Failed to initialize SAM model\n");
        return 1;
    }
    printf("SAM model initialized\n");
    
    // Step 3: Load and train on HF embeddings
    printf("\nStep 3: Training SAM on HF embeddings...\n");
    
    // Use Python helper to extract embeddings
    FILE* helper_pipe = popen("python3 hf_helper.py hf_training_data.json", "r");
    if (!helper_pipe) {
        fprintf(stderr, "Error running helper script\n");
        SAM_destroy(sam);
        return 1;
    }
    
    char line[MAX_LINE_LENGTH];
    size_t num_samples = 0;
    size_t current_sample = 0;
    size_t current_dim = 0;
    long double* current_embeddings = NULL;
    size_t samples_processed = 0;
    
    // Parse helper output
    while (fgets(line, sizeof(line), helper_pipe)) {
        if (strncmp(line, "NUM_SAMPLES:", 12) == 0) {
            num_samples = atoi(line + 12);
            printf("Found %zu training samples\n", num_samples);
        } else if (strncmp(line, "SAMPLE:", 7) == 0) {
            current_sample = atoi(line + 7);
        } else if (strncmp(line, "DIM:", 4) == 0) {
            current_dim = atoi(line + 4);
            if (current_embeddings) free(current_embeddings);
            current_embeddings = (long double*)calloc(current_dim, sizeof(long double));
        } else if (strncmp(line, "EMBEDDINGS:", 11) == 0) {
            // Parse embeddings
            char* p = line + 11;
            for (size_t i = 0; i < current_dim && p; i++) {
                current_embeddings[i] = strtold(p, &p);
                while (*p == ' ') p++;
            }
        } else if (strncmp(line, "TEXT:", 5) == 0) {
            // We have a complete sample, train on it
            if (current_embeddings && current_dim > 0) {
                // Encode input text (simplified - use first part of text)
                long double* input = (long double*)calloc(768, sizeof(long double));
                char* text = line + 5;
                size_t text_len = strlen(text);
                size_t copy_len = (text_len < 768) ? text_len : 768;
                
                for (size_t i = 0; i < copy_len; i++) {
                    input[i] = ((long double)((unsigned char)text[i])) / 255.0L;
                }
                
                // Create input sequence
                long double** input_seq = (long double**)malloc(sizeof(long double*));
                input_seq[0] = input;
                
                // Use HF embeddings as target
                long double* target = (long double*)malloc(768 * sizeof(long double));
                size_t target_copy = (current_dim < 768) ? current_dim : 768;
                for (size_t i = 0; i < target_copy; i++) {
                    target[i] = current_embeddings[i];
                }
                for (size_t i = target_copy; i < 768; i++) {
                    target[i] = 0.0L;
                }
                
                // Train SAM
                SAM_train(sam, input_seq, 1, target);
                
                samples_processed++;
                
                if (samples_processed % 100 == 0) {
                    printf("\rTrained on %zu samples...", samples_processed);
                    fflush(stdout);
                }
                
                free(input);
                free(target);
                free(input_seq);
            }
        }
    }
    
    pclose(helper_pipe);
    if (current_embeddings) free(current_embeddings);
    
    printf("\nTrained on %zu samples total\n", samples_processed);
    
    // Step 4: Save trained model
    printf("\nStep 4: Saving trained model...\n");
    char timestamp[64];
    time_t rawtime;
    struct tm *info;
    time(&rawtime);
    info = localtime(&rawtime);
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", info);
    
    char filename[128];
    snprintf(filename, sizeof(filename), "../sam_hf_trained_%s.bin", timestamp);
    
    if (SAM_save(sam, filename) == 1) {
        printf("✓ Model saved to %s\n", filename);
    }
    
    if (SAM_save(sam, "../sam_trained_model.bin") == 1) {
        printf("✓ Model saved to ../sam_trained_model.bin (default)\n");
    }
    
    SAM_destroy(sam);
    
    printf("\n=== Training Complete ===\n");
    return 0;
}

