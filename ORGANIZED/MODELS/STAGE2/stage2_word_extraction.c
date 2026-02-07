#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>

#define MAX_WORD_LENGTH 50
#define MAX_VOCABULARY_SIZE 10000
#define MIN_WORD_LENGTH 2
#define MAX_FREQUENCY 1000000

// Word entry structure
typedef struct {
    char word[MAX_WORD_LENGTH];
    int frequency;
    long double *vector;
    int vector_length;
} WordEntry;

// Vocabulary structure
typedef struct {
    WordEntry *words;
    size_t count;
    size_t max_size;
    int total_words;
} Vocabulary;

// Initialize vocabulary
Vocabulary* init_vocabulary(size_t max_size) {
    Vocabulary *vocab = malloc(sizeof(Vocabulary));
    if (!vocab) return NULL;
    
    vocab->words = malloc(max_size * sizeof(WordEntry));
    if (!vocab->words) {
        free(vocab);
        return NULL;
    }
    
    vocab->count = 0;
    vocab->max_size = max_size;
    vocab->total_words = 0;
    
    // Initialize all words
    for (size_t i = 0; i < max_size; i++) {
        vocab->words[i].word[0] = '\0';
        vocab->words[i].frequency = 0;
        vocab->words[i].vector = NULL;
        vocab->words[i].vector_length = 0;
    }
    
    return vocab;
}

// Check if a character is part of a word
bool is_word_char(char c) {
    return isalpha(c) || c == '\'' || c == '-';
}

// Extract word from text
int extract_word(const char *text, int start, char *word) {
    int i = start;
    int word_len = 0;
    
    // Skip non-word characters
    while (text[i] && !is_word_char(text[i])) {
        i++;
    }
    
    // Extract word characters
    while (text[i] && is_word_char(text[i]) && word_len < MAX_WORD_LENGTH - 1) {
        word[word_len++] = tolower(text[i]);
        i++;
    }
    
    word[word_len] = '\0';
    return i;
}

// Add word to vocabulary
int add_word_to_vocabulary(Vocabulary *vocab, const char *word) {
    // Skip words that are too short or too long
    int word_len = strlen(word);
    if (word_len < MIN_WORD_LENGTH || word_len >= MAX_WORD_LENGTH) {
        return 0;
    }
    
    // Check if word already exists
    for (size_t i = 0; i < vocab->count; i++) {
        if (strcmp(vocab->words[i].word, word) == 0) {
            vocab->words[i].frequency++;
            vocab->total_words++;
            return 1;
        }
    }
    
    // Add new word if vocabulary not full
    if (vocab->count < vocab->max_size) {
        strcpy(vocab->words[vocab->count].word, word);
        vocab->words[vocab->count].frequency = 1;
        vocab->words[vocab->count].vector = NULL;
        vocab->words[vocab->count].vector_length = 0;
        vocab->count++;
        vocab->total_words++;
        return 1;
    }
    
    return 0; // Vocabulary full
}

// Process text file and build vocabulary
int build_vocabulary_from_file(Vocabulary *vocab, const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Cannot open file %s\n", filename);
        return 0;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Read entire file
    char *text = malloc(file_size + 1);
    if (!text) {
        fclose(file);
        return 0;
    }
    
    size_t read_size = fread(text, 1, file_size, file);
    text[read_size] = '\0';
    fclose(file);
    
    printf("Processing %zu characters from %s...\n", read_size, filename);
    
    // Extract words
    int pos = 0;
    int words_added = 0;
    
    while (pos < read_size) {
        char word[MAX_WORD_LENGTH];
        pos = extract_word(text, pos, word);
        
        if (strlen(word) >= MIN_WORD_LENGTH) {
            if (add_word_to_vocabulary(vocab, word)) {
                words_added++;
            }
        }
        
        // Skip to next word
        while (pos < read_size && !is_word_char(text[pos])) {
            pos++;
        }
    }
    
    free(text);
    printf("Added %d words from %s\n", words_added, filename);
    return words_added;
}

// Sort vocabulary by frequency (descending)
void sort_vocabulary_by_frequency(Vocabulary *vocab) {
    for (size_t i = 0; i < vocab->count - 1; i++) {
        for (size_t j = i + 1; j < vocab->count; j++) {
            if (vocab->words[j].frequency > vocab->words[i].frequency) {
                // Swap
                WordEntry temp = vocab->words[i];
                vocab->words[i] = vocab->words[j];
                vocab->words[j] = temp;
            }
        }
    }
}

// Create word vectors (character-based encoding)
void create_word_vectors(Vocabulary *vocab, int vector_length) {
    printf("Creating word vectors (length: %d)...\n", vector_length);
    
    for (size_t i = 0; i < vocab->count; i++) {
        WordEntry *word = &vocab->words[i];
        
        // Allocate vector
        word->vector = malloc(vector_length * sizeof(long double));
        word->vector_length = vector_length;
        
        if (!word->vector) {
            printf("Memory allocation failed for word: %s\n", word->word);
            continue;
        }
        
        // Initialize vector with character encoding
        for (int j = 0; j < vector_length; j++) {
            if (j < strlen(word->word)) {
                // Character encoding
                char c = word->word[j];
                word->vector[j] = (long double)c / 256.0L;
            } else {
                // Padding
                word->vector[j] = 0.0L;
            }
        }
        
        // Add frequency information
        long double freq_norm = (long double)word->frequency / vocab->total_words;
        for (int j = 0; j < vector_length; j++) {
            word->vector[j] += freq_norm * 0.1L; // Small frequency influence
        }
    }
}

// Save vocabulary to file
int save_vocabulary(Vocabulary *vocab, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Error: Cannot create vocabulary file %s\n", filename);
        return 0;
    }
    
    fprintf(file, "Vocabulary\n");
    fprintf(file, "Count: %zu\n", vocab->count);
    fprintf(file, "Total words: %d\n", vocab->total_words);
    fprintf(file, "Vector length: %d\n", vocab->words[0].vector_length);
    fprintf(file, "\n");
    
    for (size_t i = 0; i < vocab->count; i++) {
        WordEntry *word = &vocab->words[i];
        fprintf(file, "%s %d ", word->word, word->frequency);
        
        // Save vector values
        for (int j = 0; j < word->vector_length; j++) {
            fprintf(file, "%.6Lf ", word->vector[j]);
        }
        fprintf(file, "\n");
    }
    
    fclose(file);
    printf("Vocabulary saved to %s\n", filename);
    return 1;
}

// Load vocabulary from file
Vocabulary* load_vocabulary(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Cannot open vocabulary file %s\n", filename);
        return NULL;
    }
    
    // Read header
    char line[1000];
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
    Vocabulary *vocab = init_vocabulary(count);
    if (!vocab) {
        fclose(file);
        return NULL;
    }
    
    vocab->total_words = total_words;
    
    // Read words
    for (size_t i = 0; i < count; i++) {
        if (!fgets(line, sizeof(line), file)) break;
        
        WordEntry *word = &vocab->words[i];
        
        // Parse word and frequency
        char word_str[MAX_WORD_LENGTH];
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
    printf("Vocabulary loaded from %s (%zu words)\n", filename, vocab->count);
    return vocab;
}

// Print vocabulary statistics
void print_vocabulary_stats(Vocabulary *vocab) {
    printf("\n=== Vocabulary Statistics ===\n");
    printf("Total words: %zu\n", vocab->count);
    printf("Total word instances: %d\n", vocab->total_words);
    printf("Vector length: %d\n", vocab->words[0].vector_length);
    
    if (vocab->count > 0) {
        printf("Most frequent words:\n");
        int show_count = vocab->count < 10 ? vocab->count : 10;
        for (int i = 0; i < show_count; i++) {
            printf("  %d. %s (%d times)\n", i + 1, vocab->words[i].word, vocab->words[i].frequency);
        }
    }
    printf("\n");
}

// Find word in vocabulary
WordEntry* find_word(Vocabulary *vocab, const char *word) {
    char lower_word[MAX_WORD_LENGTH];
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

// Cleanup vocabulary
void cleanup_vocabulary(Vocabulary *vocab) {
    if (vocab) {
        for (size_t i = 0; i < vocab->count; i++) {
            free(vocab->words[i].vector);
        }
        free(vocab->words);
        free(vocab);
    }
}

int main(int argc, char *argv[]) {
    printf("=== Stage 2: Word Extraction and Vocabulary Building ===\n\n");
    
    const char *input_file = "training_data/raw_texts/Frankenstein.txt";
    const char *vocab_file = "stage2_vocabulary.txt";
    int vector_length = 64; // Word vector dimension
    
    if (argc > 1) input_file = argv[1];
    if (argc > 2) vocab_file = argv[2];
    if (argc > 3) vector_length = atoi(argv[3]);
    
    printf("Configuration:\n");
    printf("  Input file: %s\n", input_file);
    printf("  Vocabulary file: %s\n", vocab_file);
    printf("  Vector length: %d\n", vector_length);
    printf("\n");
    
    // Initialize vocabulary
    Vocabulary *vocab = init_vocabulary(MAX_VOCABULARY_SIZE);
    if (!vocab) {
        printf("Failed to initialize vocabulary\n");
        return 1;
    }
    
    // Build vocabulary from file
    printf("Building vocabulary...\n");
    int words_added = build_vocabulary_from_file(vocab, input_file);
    
    if (words_added > 0) {
        // Sort by frequency
        sort_vocabulary_by_frequency(vocab);
        
        // Create word vectors
        create_word_vectors(vocab, vector_length);
        
        // Print statistics
        print_vocabulary_stats(vocab);
        
        // Save vocabulary
        save_vocabulary(vocab, vocab_file);
        
        // Test word finding
        printf("Testing word lookup:\n");
        char *test_words[] = {"the", "frankenstein", "monster", "life", "death"};
        for (int i = 0; i < 5; i++) {
            WordEntry *word = find_word(vocab, test_words[i]);
            if (word) {
                printf("  Found: %s (frequency: %d)\n", word->word, word->frequency);
            } else {
                printf("  Not found: %s\n", test_words[i]);
            }
        }
        
    } else {
        printf("No words were added to vocabulary\n");
    }
    
    // Cleanup
    cleanup_vocabulary(vocab);
    
    printf("\n=== Stage 2 Word Extraction Completed ===\n");
    printf("Ready for Stage 2: Word Recognition Training\n");
    
    return 0;
}
