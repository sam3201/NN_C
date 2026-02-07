#include "SAM/SAM.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <ctype.h>

#define INPUT_DIM 64      // Phrase vector dimension
#define OUTPUT_DIM 64     // Next phrase prediction
#define NUM_HEADS 8
#define EPOCHS 25
#define SAMPLES_PER_EPOCH 40
#define PHRASE_CONTEXT_SIZE 2  // Previous phrases to consider

// Phrase structure
typedef struct {
    char words[5][50];
    int word_count;
    int frequency;
    long double *vector;
    int vector_length;
} Phrase;

// Phrase database
typedef struct {
    Phrase *phrases;
    size_t count;
    int total_phrases;
} PhraseDatabase;

// Collocation structure
typedef struct {
    char word1[50];
    char word2[50];
    int frequency;
    long double association_strength;
} Collocation;

// Collocation database
typedef struct {
    Collocation *collocations;
    size_t count;
    int total_collocations;
} CollocationDatabase;

// Training sample
typedef struct {
    long double *input_vector;    // Context phrase vector
    long double *target_vector;   // Next phrase vector
    char context_phrases[PHRASE_CONTEXT_SIZE][5][50];
    char target_phrase[5][50];
    int phrase_word_counts[PHRASE_CONTEXT_SIZE];
} PhraseTrainingSample;

// Load phrase database
PhraseDatabase* load_phrase_database(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Cannot open phrase file %s\n", filename);
        return NULL;
    }
    
    char line[2000];
    fgets(line, sizeof(line), file); // "Phrase Database"
    fgets(line, sizeof(line), file); // Count line
    
    size_t count = 0;
    sscanf(line, "Count: %zu", &count);
    
    fgets(line, sizeof(line), file); // Total phrases line
    int total_phrases = 0;
    sscanf(line, "Total phrases: %d", &total_phrases);
    
    fgets(line, sizeof(line), file); // Vector length line
    int vector_length = 0;
    sscanf(line, "Vector length: %d", &vector_length);
    
    fgets(line, sizeof(line), file); // Empty line
    
    PhraseDatabase *db = malloc(sizeof(PhraseDatabase));
    if (!db) {
        fclose(file);
        return NULL;
    }
    
    db->phrases = malloc(count * sizeof(Phrase));
    if (!db->phrases) {
        free(db);
        fclose(file);
        return NULL;
    }
    
    db->count = 0;
    db->total_phrases = total_phrases;
    
    // Read phrases
    for (size_t i = 0; i < count; i++) {
        if (!fgets(line, sizeof(line), file)) break;
        
        Phrase *phrase = &db->phrases[i];
        
        // Parse word count
        char *ptr = line;
        phrase->word_count = atoi(ptr);
        ptr = strchr(ptr, ' ') + 1;
        
        // Parse words
        for (int j = 0; j < phrase->word_count; j++) {
            while (*ptr == ' ') ptr++;
            char *word_start = ptr;
            while (*ptr != ' ') ptr++;
            int word_len = ptr - word_start;
            if (word_len < 50) {
                strncpy(phrase->words[j], word_start, word_len);
                phrase->words[j][word_len] = '\0';
            }
        }
        
        // Parse frequency
        while (*ptr == ' ') ptr++;
        phrase->frequency = atoi(ptr);
        ptr = strchr(ptr, ' ') + 1;
        
        // Parse vector
        phrase->vector_length = vector_length;
        phrase->vector = malloc(vector_length * sizeof(long double));
        if (phrase->vector) {
            for (int j = 0; j < vector_length; j++) {
                while (*ptr == ' ') ptr++;
                phrase->vector[j] = strtold(ptr, &ptr);
            }
        }
        
        db->count++;
    }
    
    fclose(file);
    printf("Loaded phrase database: %zu phrases\n", db->count);
    return db;
}

// Load collocation database
CollocationDatabase* load_collocation_database(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Cannot open collocation file %s\n", filename);
        return NULL;
    }
    
    char line[500];
    fgets(line, sizeof(line), file); // "Collocation Database"
    fgets(line, sizeof(line), file); // Count line
    
    size_t count = 0;
    sscanf(line, "Count: %zu", &count);
    
    fgets(line, sizeof(line), file); // Total collocations line
    int total_collocations = 0;
    sscanf(line, "Total collocations: %d", &total_collocations);
    
    fgets(line, sizeof(line), file); // Empty line
    
    CollocationDatabase *db = malloc(sizeof(CollocationDatabase));
    if (!db) {
        fclose(file);
        return NULL;
    }
    
    db->collocations = malloc(count * sizeof(Collocation));
    if (!db->collocations) {
        free(db);
        fclose(file);
        return NULL;
    }
    
    db->count = 0;
    db->total_collocations = total_collocations;
    
    // Read collocations
    for (size_t i = 0; i < count; i++) {
        if (!fgets(line, sizeof(line), file)) break;
        
        Collocation *colloc = &db->collocations[i];
        
        // Parse collocation
        char word1[50], word2[50];
        int freq;
        long double strength;
        
        if (sscanf(line, "%s %s %d %Lf", word1, word2, &freq, &strength) == 4) {
            strcpy(colloc->word1, word1);
            strcpy(colloc->word2, word2);
            colloc->frequency = freq;
            colloc->association_strength = strength;
            db->count++;
        }
    }
    
    fclose(file);
    printf("Loaded collocation database: %zu collocations\n", db->count);
    return db;
}

// Find phrase in database
Phrase* find_phrase(PhraseDatabase *db, char words[][50], int word_count) {
    for (size_t i = 0; i < db->count; i++) {
        Phrase *phrase = &db->phrases[i];
        if (phrase->word_count == word_count) {
            bool match = true;
            for (int j = 0; j < word_count; j++) {
                if (strcmp(phrase->words[j], words[j]) != 0) {
                    match = false;
                    break;
                }
            }
            if (match) {
                return phrase;
            }
        }
    }
    return NULL;
}

// Create context vector from multiple phrases
void create_phrase_context_vector(PhraseDatabase *db, char context_phrases[][5][50], 
                                   int phrase_word_counts[], long double *context_vector) {
    // Initialize with zeros
    for (int i = 0; i < INPUT_DIM; i++) {
        context_vector[i] = 0.0L;
    }
    
    // Add vectors for context phrases
    int valid_phrases = 0;
    for (int i = 0; i < PHRASE_CONTEXT_SIZE; i++) {
        Phrase *phrase = find_phrase(db, context_phrases[i], phrase_word_counts[i]);
        if (phrase && phrase->vector) {
            // Add phrase vector to context
            for (int j = 0; j < INPUT_DIM && j < phrase->vector_length; j++) {
                context_vector[j] += phrase->vector[j];
            }
            valid_phrases++;
        }
    }
    
    // Average the vectors
    if (valid_phrases > 0) {
        for (int i = 0; i < INPUT_DIM; i++) {
            context_vector[i] /= valid_phrases;
        }
    }
}

// Create phrase training sample from text
int create_phrase_training_sample(PhraseDatabase *db, const char *text, int text_length,
                                int position, PhraseTrainingSample *sample) {
    // Extract phrases from text starting at position
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
    
    // Need at least context_size + 1 phrases (context + target)
    if (word_count < PHRASE_CONTEXT_SIZE * 2 + 1) {
        return 0; // Not enough words
    }
    
    // Create context phrases (2-word phrases)
    for (int i = 0; i < PHRASE_CONTEXT_SIZE; i++) {
        int phrase_word_count = 2; // Use 2-word phrases for context
        for (int j = 0; j < phrase_word_count; j++) {
            strcpy(sample->context_phrases[i][j], words[i * phrase_word_count + j]);
        }
        sample->phrase_word_counts[i] = phrase_word_count;
    }
    
    // Create target phrase (2-word phrase)
    int target_word_count = 2;
    for (int i = 0; i < target_word_count; i++) {
        strcpy(sample->target_phrase[i], words[PHRASE_CONTEXT_SIZE * 2 + i]);
    }
    
    // Create vectors
    create_phrase_context_vector(db, sample->context_phrases, sample->phrase_word_counts, sample->input_vector);
    
    // Get target phrase vector
    Phrase *target_phrase = find_phrase(db, sample->target_phrase, target_word_count);
    if (target_phrase && target_phrase->vector) {
        for (int i = 0; i < OUTPUT_DIM && i < target_phrase->vector_length; i++) {
            sample->target_vector[i] = target_phrase->vector[i];
        }
        return 1; // Success
    }
    
    return 0; // Target phrase not found
}

// Initialize SAM model for phrase training
SAM_t* init_phrase_model() {
    printf("Initializing SAM model for phrase grouping...\n");
    printf("  Input: %d (phrase context)\n", INPUT_DIM);
    printf("  Output: %d (next phrase)\n", OUTPUT_DIM);
    printf("  Heads: %d\n", NUM_HEADS);
    
    SAM_t *sam = SAM_init(INPUT_DIM, OUTPUT_DIM, NUM_HEADS, 0);
    if (!sam) {
        printf("Failed to initialize SAM model\n");
        return NULL;
    }
    
    printf("✓ Phrase model initialized with %zu submodels\n\n", sam->num_submodels);
    return sam;
}

// Train phrase prediction model
void train_phrase_model(SAM_t *sam, PhraseDatabase *phrase_db, const char *text_file) {
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
    printf("Starting phrase grouping training...\n");
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
            PhraseTrainingSample training_sample;
            training_sample.input_vector = malloc(INPUT_DIM * sizeof(long double));
            training_sample.target_vector = malloc(OUTPUT_DIM * sizeof(long double));
            
            if (training_sample.input_vector && training_sample.target_vector) {
                // Random position in text
                int max_pos = text_size - 300; // Leave room for words
                int pos = rand() % max_pos;
                
                if (create_phrase_training_sample(phrase_db, text, text_size, pos, &training_sample)) {
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
            snprintf(checkpoint, sizeof(checkpoint), "stage3_phrase_epoch_%d.bin", epoch);
            if (SAM_save(sam, checkpoint) == 1) {
                printf("  ✓ Checkpoint saved: %s\n", checkpoint);
            }
        }
    }
    
    // Save final model
    printf("\nSaving final phrase model...\n");
    if (SAM_save(sam, "stage3_phrase_final.bin") == 1) {
        printf("✓ Phrase model saved: stage3_phrase_final.bin\n");
    }
    
    // Training summary
    printf("\n=== Phrase Training Summary ===\n");
    printf("Total training time: %ld seconds\n", time(NULL) - start_time);
    printf("Total samples processed: %zu\n", total_samples);
    if (total_samples > 0) {
        printf("Final average loss: %.6Lf\n", total_loss / total_samples);
    }
    
    free(text);
}

// Test phrase prediction
void test_phrase_prediction(SAM_t *sam, PhraseDatabase *phrase_db) {
    printf("\n=== Phrase Prediction Test ===\n");
    
    // Test contexts
    char test_contexts[5][2][5][50] = {
        {{"the", "dark"}, {"and", "stormy"}},
        {{"i", "am"}, {"become", "death"}},
        {{"the", "monster"}, {"is", "alive"}},
        {{"life", "and"}, {"death", "itself"}},
        {{"in", "the"}, {"laboratory", "created"}}
    };
    
    int test_word_counts[5][2] = {
        {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}
    };
    
    for (int test = 0; test < 5; test++) {
        printf("Context: ");
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < test_word_counts[test][i]; j++) {
                printf("%s ", test_contexts[test][i][j]);
            }
        }
        printf("→ ");
        
        // Create context vector
        long double *context_vector = malloc(INPUT_DIM * sizeof(long double));
        create_phrase_context_vector(phrase_db, test_contexts[test], test_word_counts[test], context_vector);
        
        // Get prediction
        long double **input_seq = malloc(sizeof(long double*));
        input_seq[0] = context_vector;
        
        long double *output = SAM_forward(sam, input_seq, 1);
        if (output) {
            // Find closest phrase
            long double best_distance = 1e100;
            Phrase *best_phrase = NULL;
            
            for (size_t i = 0; i < phrase_db->count && i < 1000; i++) { // Check first 1000 phrases for speed
                Phrase *phrase = &phrase_db->phrases[i];
                if (phrase->vector) {
                    long double distance = 0.0L;
                    for (int j = 0; j < OUTPUT_DIM && j < phrase->vector_length; j++) {
                        long double diff = output[j] - phrase->vector[j];
                        distance += diff * diff;
                    }
                    
                    if (distance < best_distance) {
                        best_distance = distance;
                        best_phrase = phrase;
                    }
                }
            }
            
            if (best_phrase) {
                printf("\"");
                for (int j = 0; j < best_phrase->word_count; j++) {
                    printf("%s ", best_phrase->words[j]);
                }
                printf("\" (distance: %.6f)\n", (double)sqrt(best_distance));
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
    printf("=== Stage 3: Phrase Grouping Training ===\n\n");
    
    const char *phrase_file = "stage3_phrases.txt";
    const char *colloc_file = "stage3_collocations.txt";
    const char *text_file = "training_data/raw_texts/Frankenstein.txt";
    
    if (argc > 1) phrase_file = argv[1];
    if (argc > 2) colloc_file = argv[2];
    if (argc > 3) text_file = argv[3];
    
    printf("Configuration:\n");
    printf("  Phrase file: %s\n", phrase_file);
    printf("  Collocation file: %s\n", colloc_file);
    printf("  Training text: %s\n", text_file);
    printf("\n");
    
    // Load phrase database
    PhraseDatabase *phrase_db = load_phrase_database(phrase_file);
    if (!phrase_db) {
        printf("Failed to load phrase database\n");
        return 1;
    }
    
    // Load collocation database
    CollocationDatabase *colloc_db = load_collocation_database(colloc_file);
    if (!colloc_db) {
        printf("Failed to load collocation database\n");
        // Continue without collocations
    }
    
    // Initialize SAM model
    SAM_t *sam = init_phrase_model();
    if (!sam) {
        return 1;
    }
    
    // Train phrase model
    train_phrase_model(sam, phrase_db, text_file);
    
    // Test phrase prediction
    test_phrase_prediction(sam, phrase_db);
    
    // Cleanup
    SAM_destroy(sam);
    
    // Free phrase database
    for (size_t i = 0; i < phrase_db->count; i++) {
        free(phrase_db->phrases[i].vector);
    }
    free(phrase_db->phrases);
    free(phrase_db);
    
    // Free collocation database
    if (colloc_db) {
        free(colloc_db->collocations);
        free(colloc_db);
    }
    
    printf("\n=== Stage 3 Phrase Training Completed ===\n");
    printf("Ready for Stage 4: Response Generation Training\n");
    
    return 0;
}
