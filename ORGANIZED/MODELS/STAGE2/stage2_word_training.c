#include "SAM/SAM.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <ctype.h>

#define INPUT_DIM 64      // Word vector dimension
#define OUTPUT_DIM 64     // Next word prediction
#define NUM_HEADS 8
#define EPOCHS 30
#define SAMPLES_PER_EPOCH 50
#define WORD_CONTEXT_SIZE 3  // Previous words to consider

// Word entry structure (matching extraction)
typedef struct {
    char word[50];
    int frequency;
    long double *vector;
    int vector_length;
} WordEntry;

// Vocabulary structure
typedef struct {
    WordEntry *words;
    size_t count;
    int total_words;
} Vocabulary;

// Training sample
typedef struct {
    long double *input_vector;    // Context words vector
    long double *target_vector;   // Next word vector
    char context_words[WORD_CONTEXT_SIZE][50];
    char target_word[50];
} WordTrainingSample;

// Load vocabulary from file
Vocabulary* load_vocabulary(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Cannot open vocabulary file %s\n", filename);
        return NULL;
    }
    
    // Read header
    char line[2000];
    fgets(line, sizeof(line), file); // "Vocabulary"
    fgets(line, sizeof(line), file); // Count line
    
    size_t count = 0;
    sscanf(line, "Count: %zu", &count);
    
    fgets(line, sizeof(line), file); // Total words line
    int total_words = 0;
    sscanf(line, "Total words: %d", &total_words);
    
    fgets(line, sizeof(line), file); // Vector length line
    int vector_length = 0;
    sscanf(line, "Vector length: %d", &vector_length);
    
    fgets(line, sizeof(line), file); // Empty line
    
    // Initialize vocabulary
    Vocabulary *vocab = malloc(sizeof(Vocabulary));
    if (!vocab) {
        fclose(file);
        return NULL;
    }
    
    vocab->words = malloc(count * sizeof(WordEntry));
    if (!vocab->words) {
        free(vocab);
        fclose(file);
        return NULL;
    }
    
    vocab->count = 0;
    vocab->total_words = total_words;
    
    // Read words
    for (size_t i = 0; i < count; i++) {
        if (!fgets(line, sizeof(line), file)) break;
        
        WordEntry *word = &vocab->words[i];
        
        // Parse word and frequency
        char word_str[50];
        int freq;
        int parsed = sscanf(line, "%s %d", word_str, &freq);
        
        if (parsed >= 2) {
            strcpy(word->word, word_str);
            word->frequency = freq;
            word->vector_length = vector_length;
            
            // Allocate and read vector
            word->vector = malloc(vector_length * sizeof(long double));
            if (word->vector) {
                char *ptr = line;
                // Skip word and frequency
                ptr = strchr(ptr, ' ') + 1;
                ptr = strchr(ptr, ' ') + 1;
                
                // Read vector values
                for (int j = 0; j < vector_length; j++) {
                    while (*ptr == ' ') ptr++;
                    word->vector[j] = strtold(ptr, &ptr);
                }
            }
            
            vocab->count++;
        }
    }
    
    fclose(file);
    printf("Loaded vocabulary: %zu words\n", vocab->count);
    return vocab;
}

// Find word in vocabulary
WordEntry* find_word(Vocabulary *vocab, const char *word) {
    char lower_word[50];
    strcpy(lower_word, word);
    
    // Convert to lowercase
    for (int i = 0; lower_word[i]; i++) {
        lower_word[i] = tolower(lower_word[i]);
    }
    
    for (size_t i = 0; i < vocab->count; i++) {
        if (strcmp(vocab->words[i].word, lower_word) == 0) {
            return &vocab->words[i];
        }
    }
    
    return NULL; // Word not found
}

// Create context vector from multiple words
void create_context_vector(Vocabulary *vocab, char context_words[][50], 
                           int context_count, long double *context_vector) {
    // Initialize with zeros
    for (int i = 0; i < INPUT_DIM; i++) {
        context_vector[i] = 0.0L;
    }
    
    // Add vectors for context words
    int valid_words = 0;
    for (int i = 0; i < context_count; i++) {
        WordEntry *word = find_word(vocab, context_words[i]);
        if (word && word->vector) {
            // Add word vector to context
            for (int j = 0; j < INPUT_DIM && j < word->vector_length; j++) {
                context_vector[j] += word->vector[j];
            }
            valid_words++;
        }
    }
    
    // Average the vectors
    if (valid_words > 0) {
        for (int i = 0; i < INPUT_DIM; i++) {
            context_vector[i] /= valid_words;
        }
    }
}

// Create training sample from text
int create_training_sample(Vocabulary *vocab, const char *text, int text_length,
                          int position, WordTrainingSample *sample) {
    // Extract words from text starting at position
    char words[20][50]; // Up to 20 words in context
    int word_count = 0;
    
    int pos = position;
    while (pos < text_length && word_count < 20) {
        // Skip non-word characters
        while (pos < text_length && !(isalpha(text[pos]) || text[pos] == '\'' || text[pos] == '-')) {
            pos++;
        }
        
        if (pos >= text_length) break;
        
        // Extract word
        int word_start = pos;
        while (pos < text_length && (isalpha(text[pos]) || text[pos] == '\'' || text[pos] == '-')) {
            pos++;
        }
        
        int word_len = pos - word_start;
        if (word_len > 0 && word_len < 50) {
            strncpy(words[word_count], text + word_start, word_len);
            words[word_count][word_len] = '\0';
            
            // Convert to lowercase
            for (int i = 0; i < word_len; i++) {
                words[word_count][i] = tolower(words[word_count][i]);
            }
            
            word_count++;
        }
    }
    
    // Need at least context_size + 1 words (context + target)
    if (word_count < WORD_CONTEXT_SIZE + 1) {
        return 0; // Not enough words
    }
    
    // Set context words
    for (int i = 0; i < WORD_CONTEXT_SIZE; i++) {
        strcpy(sample->context_words[i], words[i]);
    }
    
    // Set target word
    strcpy(sample->target_word, words[WORD_CONTEXT_SIZE]);
    
    // Create vectors
    create_context_vector(vocab, sample->context_words, WORD_CONTEXT_SIZE, sample->input_vector);
    
    // Get target word vector
    WordEntry *target_word_entry = find_word(vocab, sample->target_word);
    if (target_word_entry && target_word_entry->vector) {
        for (int i = 0; i < OUTPUT_DIM && i < target_word_entry->vector_length; i++) {
            sample->target_vector[i] = target_word_entry->vector[i];
        }
        return 1; // Success
    }
    
    return 0; // Target word not found
}

// Initialize SAM model for word training
SAM_t* init_word_model() {
    printf("Initializing SAM model for word recognition...\n");
    printf("  Input: %d (word context)\n", INPUT_DIM);
    printf("  Output: %d (next word)\n", OUTPUT_DIM);
    printf("  Heads: %d\n", NUM_HEADS);
    
    SAM_t *sam = SAM_init(INPUT_DIM, OUTPUT_DIM, NUM_HEADS, 0);
    if (!sam) {
        printf("Failed to initialize SAM model\n");
        return NULL;
    }
    
    printf("✓ Word model initialized with %zu submodels\n\n", sam->num_submodels);
    return sam;
}

// Train word prediction model
void train_word_model(SAM_t *sam, Vocabulary *vocab, const char *text_file) {
    printf("Loading training text: %s\n", text_file);
    
    FILE *file = fopen(text_file, "r");
    if (!file) {
        printf("Error: Cannot open text file %s\n", text_file);
        return;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Read text
    char *text = malloc(file_size + 1);
    if (!text) {
        fclose(file);
        return;
    }
    
    size_t text_size = fread(text, 1, file_size, file);
    text[text_size] = '\0';
    fclose(file);
    
    printf("Loaded %zu characters of training text\n", text_size);
    
    // Training loop
    printf("Starting word recognition training...\n");
    printf("Epochs: %d, Samples per epoch: %d\n\n", EPOCHS, SAMPLES_PER_EPOCH);
    
    time_t start_time = time(NULL);
    long double total_loss = 0.0L;
    size_t total_samples = 0;
    
    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        long double epoch_loss = 0.0L;
        size_t epoch_samples = 0;
        
        printf("Epoch %d/%d - ", epoch, EPOCHS);
        
        for (int sample = 0; sample < SAMPLES_PER_EPOCH; sample++) {
            // Create training sample
            WordTrainingSample training_sample;
            training_sample.input_vector = malloc(INPUT_DIM * sizeof(long double));
            training_sample.target_vector = malloc(OUTPUT_DIM * sizeof(long double));
            
            if (training_sample.input_vector && training_sample.target_vector) {
                // Random position in text
                int max_pos = text_size - 200; // Leave room for words
                int pos = rand() % max_pos;
                
                if (create_training_sample(vocab, text, text_size, pos, &training_sample)) {
                    // Create input sequence
                    long double **input_seq = malloc(sizeof(long double*));
                    input_seq[0] = training_sample.input_vector;
                    
                    // Train model
                    SAM_train(sam, input_seq, 1, training_sample.target_vector);
                    
                    // Calculate loss
                    long double *output = SAM_forward(sam, input_seq, 1);
                    if (output) {
                        long double sample_loss = 0.0L;
                        for (int i = 0; i < OUTPUT_DIM; i++) {
                            long double error = output[i] - training_sample.target_vector[i];
                            sample_loss += error * error;
                        }
                        epoch_loss += sample_loss / OUTPUT_DIM;
                        epoch_samples++;
                        total_loss += sample_loss / OUTPUT_DIM;
                        total_samples++;
                        free(output);
                    }
                    
                    // Adapt model
                    SAM_adapt(sam, input_seq, 1);
                    
                    free(input_seq);
                }
            }
            
            free(training_sample.input_vector);
            free(training_sample.target_vector);
        }
        
        time_t elapsed = time(NULL) - start_time;
        long double avg_loss = epoch_samples > 0 ? epoch_loss / epoch_samples : 0.0L;
        
        printf("Loss: %.6Lf - Samples: %zu - Time: %lds\n", avg_loss, epoch_samples, elapsed);
        
        // Save checkpoint every 5 epochs
        if (epoch % 5 == 0) {
            char checkpoint[100];
            snprintf(checkpoint, sizeof(checkpoint), "stage2_word_epoch_%d.bin", epoch);
            if (SAM_save(sam, checkpoint) == 1) {
                printf("  ✓ Checkpoint saved: %s\n", checkpoint);
            }
        }
    }
    
    // Save final model
    printf("\nSaving final word model...\n");
    if (SAM_save(sam, "stage2_word_final.bin") == 1) {
        printf("✓ Word model saved: stage2_word_final.bin\n");
    }
    
    // Training summary
    printf("\n=== Word Training Summary ===\n");
    printf("Total training time: %ld seconds\n", time(NULL) - start_time);
    printf("Total samples processed: %zu\n", total_samples);
    if (total_samples > 0) {
        printf("Final average loss: %.6Lf\n", total_loss / total_samples);
    }
    
    free(text);
}

// Test word prediction
void test_word_prediction(SAM_t *sam, Vocabulary *vocab) {
    printf("\n=== Word Prediction Test ===\n");
    
    // Test contexts
    char test_contexts[5][WORD_CONTEXT_SIZE][50] = {
        {"the", "dark", "and"},
        {"i", "am", "become"},
        {"the", "monster", "is"},
        {"life", "and", "death"},
        {"in", "the", "laboratory"}
    };
    
    for (int test = 0; test < 5; test++) {
        printf("Context: ");
        for (int i = 0; i < WORD_CONTEXT_SIZE; i++) {
            printf("%s ", test_contexts[test][i]);
        }
        printf("→ ");
        
        // Create context vector
        long double *context_vector = malloc(INPUT_DIM * sizeof(long double));
        create_context_vector(vocab, test_contexts[test], WORD_CONTEXT_SIZE, context_vector);
        
        // Get prediction
        long double **input_seq = malloc(sizeof(long double*));
        input_seq[0] = context_vector;
        
        long double *output = SAM_forward(sam, input_seq, 1);
        if (output) {
            // Find closest word in vocabulary
            long double best_distance = 1e100;
            WordEntry *best_word = NULL;
            
            for (size_t i = 0; i < vocab->count && i < 1000; i++) { // Check first 1000 words for speed
                WordEntry *word = &vocab->words[i];
                if (word->vector) {
                    long double distance = 0.0L;
                    for (int j = 0; j < OUTPUT_DIM && j < word->vector_length; j++) {
                        long double diff = output[j] - word->vector[j];
                        distance += diff * diff;
                    }
                    
                    if (distance < best_distance) {
                        best_distance = distance;
                        best_word = word;
                    }
                }
            }
            
            if (best_word) {
                printf("\"%s\" (distance: %.6f)\n", best_word->word, (double)sqrt(best_distance));
            } else {
                printf("No prediction found\n");
            }
            
            free(output);
        }
        
        free(input_seq);
        free(context_vector);
    }
}

int main(int argc, char *argv[]) {
    printf("=== Stage 2: Word Recognition Training ===\n\n");
    
    const char *vocab_file = "stage2_vocabulary.txt";
    const char *text_file = "training_data/raw_texts/Frankenstein.txt";
    
    if (argc > 1) vocab_file = argv[1];
    if (argc > 2) text_file = argv[2];
    
    printf("Configuration:\n");
    printf("  Vocabulary file: %s\n", vocab_file);
    printf("  Training text: %s\n", text_file);
    printf("\n");
    
    // Load vocabulary
    Vocabulary *vocab = load_vocabulary(vocab_file);
    if (!vocab) {
        printf("Failed to load vocabulary\n");
        return 1;
    }
    
    // Initialize SAM model
    SAM_t *sam = init_word_model();
    if (!sam) {
        // Free vocabulary
        for (size_t i = 0; i < vocab->count; i++) {
            free(vocab->words[i].vector);
        }
        free(vocab->words);
        free(vocab);
        return 1;
    }
    
    // Train word model
    train_word_model(sam, vocab, text_file);
    
    // Test word prediction
    test_word_prediction(sam, vocab);
    
    // Cleanup
    SAM_destroy(sam);
    
    // Free vocabulary
    for (size_t i = 0; i < vocab->count; i++) {
        free(vocab->words[i].vector);
    }
    free(vocab->words);
    free(vocab);
    
    printf("\n=== Stage 2 Word Training Completed ===\n");
    printf("Ready for Stage 3: Word Grouping Training\n");
    
    return 0;
}
