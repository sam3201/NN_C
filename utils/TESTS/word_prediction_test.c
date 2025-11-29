#include "../NN/transformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>  // For LDBL_MAX

#define MAX_WORD_LENGTH 50
#define VOCAB_SIZE 256  // Using ASCII encoding for simplicity
#define SEQUENCE_LENGTH 10
#define MODEL_DIM 64
#define NUM_HEADS 4
#define FF_DIM 256
#define BATCH_SIZE 32

// Structure to hold our vocabulary and embeddings
typedef struct {
    char** words;
    size_t num_words;
    long double** embeddings;  // One embedding vector per word
} Vocabulary;

// Function to load words from file
Vocabulary* load_vocabulary(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return NULL;
    }

    Vocabulary* vocab = (Vocabulary*)malloc(sizeof(Vocabulary));
    if (!vocab) {
        fclose(file);
        return NULL;
    }

    // Initialize with some capacity
    size_t capacity = 1000;
    vocab->words = (char**)malloc(capacity * sizeof(char*));
    vocab->embeddings = (long double**)malloc(capacity * sizeof(long double*));
    vocab->num_words = 0;

    char word[MAX_WORD_LENGTH];
    while (fscanf(file, "%s", word) == 1) {
        if (vocab->num_words >= capacity) {
            capacity *= 2;
            char** new_words = (char**)realloc(vocab->words, capacity * sizeof(char*));
            long double** new_embeddings = (long double**)realloc(vocab->embeddings, capacity * sizeof(long double*));
            if (!new_words || !new_embeddings) {
                fprintf(stderr, "Failed to reallocate memory\n");
                // Clean up
                for (size_t i = 0; i < vocab->num_words; i++) {
                    free(vocab->words[i]);
                    free(vocab->embeddings[i]);
                }
                free(vocab->words);
                free(vocab->embeddings);
                free(vocab);
                fclose(file);
                return NULL;
            }
            vocab->words = new_words;
            vocab->embeddings = new_embeddings;
        }

        vocab->words[vocab->num_words] = strdup(word);
        vocab->embeddings[vocab->num_words] = (long double*)calloc(MODEL_DIM, sizeof(long double));
        
        // Initialize embedding with random values
        for (size_t i = 0; i < MODEL_DIM; i++) {
            vocab->embeddings[vocab->num_words][i] = ((long double)rand() / RAND_MAX * 2.0L - 1.0L) * 0.1L;
        }
        
        vocab->num_words++;
    }

    fclose(file);
    return vocab;
}

// Function to free vocabulary
void free_vocabulary(Vocabulary* vocab) {
    if (!vocab) return;
    
    for (size_t i = 0; i < vocab->num_words; i++) {
        free(vocab->words[i]);
        free(vocab->embeddings[i]);
    }
    free(vocab->words);
    free(vocab->embeddings);
    free(vocab);
}

// Function to encode a word into a one-hot vector
void encode_word(const char* word, long double* encoded, size_t dim) {
    memset(encoded, 0, dim * sizeof(long double));
    size_t len = strlen(word);
    for (size_t i = 0; i < len && i < dim; i++) {
        encoded[i] = ((long double)word[i]) / 255.0L;  // Normalize to [0,1]
    }
}

// Function to decode a vector back into a word
void decode_vector(const long double* vec, size_t dim, char* word) {
    size_t j = 0;
    for (size_t i = 0; i < dim && j < MAX_WORD_LENGTH - 1; i++) {
        if (vec[i] > 0.1L) {  // Threshold
            word[j++] = (char)(vec[i] * 255.0L);
        }
    }
    word[j] = '\0';
}

// Function to find the nearest word in vocabulary
const char* find_nearest_word(const Vocabulary* vocab, const long double* vec) {
    if (!vocab || !vec) return NULL;
    
    size_t nearest_idx = 0;
    long double min_dist = LDBL_MAX;
    
    for (size_t i = 0; i < vocab->num_words; i++) {
        long double dist = 0;
        for (size_t j = 0; j < MODEL_DIM; j++) {
            long double diff = vocab->embeddings[i][j] - vec[j];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            nearest_idx = i;
        }
    }
    
    return vocab->words[nearest_idx];
}

// Function to train the transformer on word prediction
void train_word_prediction(TransformerLayer* layer, Vocabulary* vocab, size_t num_epochs) {
    if (!layer || !vocab || vocab->num_words < 2) return;
    
    long double* input = (long double*)malloc(MODEL_DIM * sizeof(long double));
    long double* target = (long double*)malloc(MODEL_DIM * sizeof(long double));
    
    printf("Training for %zu epochs...\n", num_epochs);
    
    for (size_t epoch = 0; epoch < num_epochs; epoch++) {
        long double total_loss = 0;
        size_t num_samples = 0;
        
        // Train on consecutive word pairs
        for (size_t i = 0; i < vocab->num_words - 1; i++) {
            // Encode input word
            encode_word(vocab->words[i], input, MODEL_DIM);
            
            // Encode target (next word)
            encode_word(vocab->words[i + 1], target, MODEL_DIM);
            
            // Forward pass
            long double* output = transformer_forward(layer, input);
            if (!output) continue;
            
            // Compute loss (MSE)
            long double loss = 0;
            for (size_t j = 0; j < MODEL_DIM; j++) {
                long double diff = output[j] - target[j];
                loss += diff * diff;
            }
            loss /= MODEL_DIM;
            total_loss += loss;
            num_samples++;
            
            // Update embeddings (simple gradient descent)
            const long double learning_rate = 0.01L;
            for (size_t j = 0; j < MODEL_DIM; j++) {
                vocab->embeddings[i][j] -= learning_rate * (output[j] - target[j]);
            }
            
            free(output);
            
            // Print progress every 1000 samples
            if (num_samples % 1000 == 0) {
                printf("\rEpoch %zu, Samples: %zu, Avg Loss: %.6Lf", 
                       epoch + 1, num_samples, total_loss / num_samples);
                fflush(stdout);
            }
        }
        
        printf("\nEpoch %zu completed. Average loss: %.6Lf\n", 
               epoch + 1, total_loss / num_samples);
    }
    
    free(input);
    free(target);
}

// Function to encode text into sequences for training
void create_training_data(const char* input_file, const char* output_file, size_t sequence_length) {
    FILE* fin = fopen(input_file, "r");
    FILE* fout = fopen(output_file, "w");
    if (!fin || !fout) {
        fprintf(stderr, "Error opening files\n");
        return;
    }

    // Buffer for reading text
    char* line = NULL;
    size_t len = 0;
    ssize_t read;

    // Buffer for words in a sequence
    char** word_sequence = (char**)malloc(sequence_length * sizeof(char*));
    for (size_t i = 0; i < sequence_length; i++) {
        word_sequence[i] = (char*)malloc(MAX_WORD_LENGTH * sizeof(char));
    }
    size_t sequence_pos = 0;

    // Write header for the output CSV
    fprintf(fout, "sequence,next_word\n");

    // Read file line by line
    while ((read = getline(&line, &len, fin)) != -1) {
        char* word = strtok(line, " \t\n");
        while (word) {
            // Add word to current sequence
            strncpy(word_sequence[sequence_pos], word, MAX_WORD_LENGTH - 1);
            word_sequence[sequence_pos][MAX_WORD_LENGTH - 1] = '\0';
            sequence_pos++;

            // If we have a complete sequence
            if (sequence_pos == sequence_length) {
                // Get next word for target
                word = strtok(NULL, " \t\n");
                if (word) {
                    // Write sequence and target to CSV
                    for (size_t i = 0; i < sequence_length; i++) {
                        fprintf(fout, "%s", word_sequence[i]);
                        if (i < sequence_length - 1) fprintf(fout, " ");
                    }
                    fprintf(fout, ",%s\n", word);

                    // Shift sequence left by one
                    for (size_t i = 0; i < sequence_length - 1; i++) {
                        strcpy(word_sequence[i], word_sequence[i + 1]);
                    }
                    sequence_pos--;
                }
            }
            word = strtok(NULL, " \t\n");
        }
    }

    // Clean up
    for (size_t i = 0; i < sequence_length; i++) {
        free(word_sequence[i]);
    }
    free(word_sequence);
    free(line);
    fclose(fin);
    fclose(fout);
}

// Structure to hold training data
typedef struct {
    char*** sequences;  // Array of word sequences
    char** targets;     // Array of target words
    size_t num_samples; // Number of training samples
    size_t seq_length;  // Length of each sequence
} TrainingData;

// Function to load training data
TrainingData* load_training_data(const char* filename, size_t sequence_length) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error opening training data file\n");
        return NULL;
    }

    TrainingData* data = (TrainingData*)malloc(sizeof(TrainingData));
    data->seq_length = sequence_length;
    data->num_samples = 0;
    data->sequences = NULL;
    data->targets = NULL;

    // Skip header
    char* line = NULL;
    size_t len = 0;
    getline(&line, &len, f);
    free(line);
    line = NULL;

    // Count lines first
    while (getline(&line, &len, f) != -1) {
        data->num_samples++;
        free(line);
        line = NULL;
    }
    rewind(f);

    // Skip header again
    getline(&line, &len, f);
    free(line);
    line = NULL;

    // Allocate memory
    data->sequences = (char***)malloc(data->num_samples * sizeof(char**));
    for (size_t i = 0; i < data->num_samples; i++) {
        data->sequences[i] = (char**)malloc(sequence_length * sizeof(char*));
        for (size_t j = 0; j < sequence_length; j++) {
            data->sequences[i][j] = (char*)malloc(MAX_WORD_LENGTH * sizeof(char));
        }
    }
    data->targets = (char**)malloc(data->num_samples * sizeof(char*));
    for (size_t i = 0; i < data->num_samples; i++) {
        data->targets[i] = (char*)malloc(MAX_WORD_LENGTH * sizeof(char));
    }

    // Read data
    size_t sample_idx = 0;
    while (getline(&line, &len, f) != -1) {
        char* sequence_str = strtok(line, ",");
        char* target = strtok(NULL, "\n");
        
        // Parse sequence
        char* word = strtok(sequence_str, " ");
        for (size_t i = 0; i < sequence_length; i++) {
            if (word) {
                strncpy(data->sequences[sample_idx][i], word, MAX_WORD_LENGTH - 1);
                data->sequences[sample_idx][i][MAX_WORD_LENGTH - 1] = '\0';
                word = strtok(NULL, " ");
            }
        }
        
        // Store target
        strncpy(data->targets[sample_idx], target, MAX_WORD_LENGTH - 1);
        data->targets[sample_idx][MAX_WORD_LENGTH - 1] = '\0';
        
        sample_idx++;
    }

    fclose(f);
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

void test_word_prediction() {
    printf("Creating training data...\n");
    create_training_data("input.txt", "training_data.csv", SEQUENCE_LENGTH);
    
    printf("Loading training data...\n");
    TrainingData* train_data = load_training_data("training_data.csv", SEQUENCE_LENGTH);
    if (!train_data) {
        fprintf(stderr, "Failed to load training data\n");
        return;
    }
    printf("Loaded %zu training samples\n", train_data->num_samples);

    printf("Loading vocabulary...\n");
    Vocabulary* vocab = load_vocabulary("words.txt");
    if (!vocab) {
        fprintf(stderr, "Failed to load vocabulary\n");
        free_training_data(train_data);
        return;
    }
    printf("Loaded %zu words\n", vocab->num_words);

    // Create transformer layer
    TransformerLayer* layer = create_transformer_layer(MODEL_DIM, NUM_HEADS, FF_DIM);
    if (!layer) {
        fprintf(stderr, "Failed to create transformer layer\n");
        free_vocabulary(vocab);
        free_training_data(train_data);
        return;
    }

    // Train the model
    printf("\nTraining model...\n");
    const size_t num_epochs = 5;
    const long double learning_rate = 0.01L;
    
    for (size_t epoch = 0; epoch < num_epochs; epoch++) {
        long double epoch_loss = 0;
        
        for (size_t i = 0; i < train_data->num_samples; i++) {
            // Encode input sequence
            long double* input = (long double*)calloc(MODEL_DIM, sizeof(long double));
            for (size_t j = 0; j < train_data->seq_length; j++) {
                encode_word(train_data->sequences[i][j], input, MODEL_DIM);
            }
            
            // Encode target
            long double* target = (long double*)calloc(MODEL_DIM, sizeof(long double));
            encode_word(train_data->targets[i], target, MODEL_DIM);
                        
            // Forward pass
            long double* output = transformer_forward(layer, input);

            // Calculate loss and gradients (MSE)
            long double* grad_output = (long double*)malloc(MODEL_DIM * sizeof(long double));
            long double sample_loss = 0;
            for (size_t j = 0; j < MODEL_DIM; j++) {
                long double diff = output[j] - target[j];
                sample_loss += diff * diff;
                grad_output[j] = 2 * diff;  // d/dy ( (y - t)^2 ) = 2 (y - t)
            }
            epoch_loss += sample_loss / MODEL_DIM;

            // Backpropagate through transformer
            long double* grad_input = (long double*)malloc(MODEL_DIM * sizeof(long double));
            if (grad_input) {
                transformer_backprop(layer, input, grad_output, grad_input);
                free(grad_input);
            }

            // Clean up
            free(input);
            free(target);
            free(output);
            free(grad_output);
        }
    }
    
    // Test prediction
    printf("\nTesting word prediction...\n");
    const char* test_sequences[][SEQUENCE_LENGTH] = {
        {"the", "quick", "brown"},
        {"a", "beautiful", "summer"},
        {"in", "the", "morning"}
    };
    size_t num_tests = sizeof(test_sequences) / sizeof(test_sequences[0]);

    for (size_t i = 0; i < num_tests; i++) {
        printf("\nInput sequence: ");
        for (size_t j = 0; j < SEQUENCE_LENGTH; j++) {
            printf("%s ", test_sequences[i][j]);
        }
        printf("\n");

        // Encode input sequence
        long double* input = (long double*)calloc(MODEL_DIM, sizeof(long double));
        for (size_t j = 0; j < SEQUENCE_LENGTH; j++) {
            encode_word(test_sequences[i][j], input, MODEL_DIM);
        }

        // Forward pass
        long double* output = transformer_forward(layer, input);
        if (!output) {
            fprintf(stderr, "Failed in transformer forward pass\n");
            continue;
        }

        // Find nearest word in vocabulary
        const char* predicted = find_nearest_word(vocab, output);
        if (predicted) {
            printf("Predicted next word: %s\n", predicted);
        }

        free(input);
        free(output);
    }

    // Clean up
    free_transformer_layer(layer);
    free_vocabulary(vocab);
    free_training_data(train_data);
}

int main() {
    test_word_prediction();
    return 0;
}
