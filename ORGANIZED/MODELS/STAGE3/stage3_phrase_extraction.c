#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>
#include <math.h>

#define MAX_PHRASE_LENGTH 5
#define MAX_PHRASES 50000
#define MIN_PHRASE_FREQUENCY 2
#define MAX_WORD_LENGTH 50

// Phrase structure
typedef struct {
    char words[MAX_PHRASE_LENGTH][MAX_WORD_LENGTH];
    int word_count;
    int frequency;
    long double *vector;
    int vector_length;
} Phrase;

// Collocation structure (word pairs)
typedef struct {
    char word1[MAX_WORD_LENGTH];
    char word2[MAX_WORD_LENGTH];
    int frequency;
    long double association_strength;
} Collocation;

// Phrase database
typedef struct {
    Phrase *phrases;
    size_t count;
    size_t max_size;
    int total_phrases;
} PhraseDatabase;

// Collocation database
typedef struct {
    Collocation *collocations;
    size_t count;
    size_t max_size;
    int total_collocations;
} CollocationDatabase;

// Word entry structure (for vocabulary lookup)
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
    int total_words;
} Vocabulary;

// Initialize phrase database
PhraseDatabase* init_phrase_database(size_t max_size) {
    PhraseDatabase *db = malloc(sizeof(PhraseDatabase));
    if (!db) return NULL;
    
    db->phrases = malloc(max_size * sizeof(Phrase));
    if (!db->phrases) {
        free(db);
        return NULL;
    }
    
    db->count = 0;
    db->max_size = max_size;
    db->total_phrases = 0;
    
    // Initialize all phrases
    for (size_t i = 0; i < max_size; i++) {
        db->phrases[i].word_count = 0;
        db->phrases[i].frequency = 0;
        db->phrases[i].vector = NULL;
        db->phrases[i].vector_length = 0;
        for (int j = 0; j < MAX_PHRASE_LENGTH; j++) {
            db->phrases[i].words[j][0] = '\0';
        }
    }
    
    return db;
}

// Initialize collocation database
CollocationDatabase* init_collocation_database(size_t max_size) {
    CollocationDatabase *db = malloc(sizeof(CollocationDatabase));
    if (!db) return NULL;
    
    db->collocations = malloc(max_size * sizeof(Collocation));
    if (!db->collocations) {
        free(db);
        return NULL;
    }
    
    db->count = 0;
    db->max_size = max_size;
    db->total_collocations = 0;
    
    // Initialize all collocations
    for (size_t i = 0; i < max_size; i++) {
        db->collocations[i].word1[0] = '\0';
        db->collocations[i].word2[0] = '\0';
        db->collocations[i].frequency = 0;
        db->collocations[i].association_strength = 0.0L;
    }
    
    return db;
}

// Load vocabulary
Vocabulary* load_vocabulary(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) return NULL;
    
    char line[2000];
    fgets(line, sizeof(line), file); // "Vocabulary"
    fgets(line, sizeof(line), file); // Count line
    
    size_t count = 0;
    sscanf(line, "Count: %zu", &count);
    
    fgets(line, sizeof(line), file); // Total words
    int total_words = 0;
    sscanf(line, "Total words: %d", &total_words);
    
    fgets(line, sizeof(line), file); // Vector length
    int vector_length = 0;
    sscanf(line, "Vector length: %d", &vector_length);
    
    fgets(line, sizeof(line), file); // Empty line
    
    Vocabulary *vocab = malloc(sizeof(Vocabulary));
    vocab->words = malloc(count * sizeof(WordEntry));
    vocab->count = 0;
    vocab->total_words = total_words;
    
    for (size_t i = 0; i < count; i++) {
        if (!fgets(line, sizeof(line), file)) break;
        
        WordEntry *word = &vocab->words[i];
        char word_str[50];
        int freq;
        
        if (sscanf(line, "%s %d", word_str, &freq) >= 2) {
            strcpy(word->word, word_str);
            word->frequency = freq;
            word->vector_length = vector_length;
            word->vector = malloc(vector_length * sizeof(long double));
            
            if (word->vector) {
                char *ptr = line;
                ptr = strchr(ptr, ' ') + 1;
                ptr = strchr(ptr, ' ') + 1;
                
                for (int j = 0; j < vector_length; j++) {
                    while (*ptr == ' ') ptr++;
                    word->vector[j] = strtold(ptr, &ptr);
                }
            }
            vocab->count++;
        }
    }
    
    fclose(file);
    return vocab;
}

// Find word in vocabulary
WordEntry* find_word(Vocabulary *vocab, const char *word) {
    char lower_word[50];
    strcpy(lower_word, word);
    for (int i = 0; lower_word[i]; i++) {
        lower_word[i] = tolower(lower_word[i]);
    }
    
    for (size_t i = 0; i < vocab->count; i++) {
        if (strcmp(vocab->words[i].word, lower_word) == 0) {
            return &vocab->words[i];
        }
    }
    return NULL;
}

// Check if character is part of a word
bool is_word_char(char c) {
    return isalpha(c) || c == '\'' || c == '-';
}

// Extract words from text
int extract_words_from_text(const char *text, int start_pos, char words[][MAX_WORD_LENGTH], int max_words) {
    int pos = start_pos;
    int word_count = 0;
    
    while (word_count < max_words) {
        // Skip non-word characters
        while (text[pos] && !is_word_char(text[pos])) {
            pos++;
        }
        
        if (!text[pos]) break;
        
        // Extract word
        int word_start = pos;
        while (text[pos] && is_word_char(text[pos])) {
            pos++;
        }
        
        int word_len = pos - word_start;
        if (word_len > 0 && word_len < MAX_WORD_LENGTH) {
            strncpy(words[word_count], text + word_start, word_len);
            words[word_count][word_len] = '\0';
            
            // Convert to lowercase
            for (int i = 0; i < word_len; i++) {
                words[word_count][i] = tolower(words[word_count][i]);
            }
            
            word_count++;
        }
    }
    
    return word_count;
}

// Add phrase to database
int add_phrase_to_database(PhraseDatabase *db, char words[][MAX_WORD_LENGTH], int word_count) {
    // Skip if too short or too long
    if (word_count < 2 || word_count > MAX_PHRASE_LENGTH) {
        return 0;
    }
    
    // Check if phrase already exists
    for (size_t i = 0; i < db->count; i++) {
        if (db->phrases[i].word_count == word_count) {
            bool match = true;
            for (int j = 0; j < word_count; j++) {
                if (strcmp(db->phrases[i].words[j], words[j]) != 0) {
                    match = false;
                    break;
                }
            }
            if (match) {
                db->phrases[i].frequency++;
                db->total_phrases++;
                return 1;
            }
        }
    }
    
    // Add new phrase if database not full
    if (db->count < db->max_size) {
        Phrase *phrase = &db->phrases[db->count];
        phrase->word_count = word_count;
        phrase->frequency = 1;
        phrase->vector = NULL;
        phrase->vector_length = 0;
        
        for (int i = 0; i < word_count; i++) {
            strcpy(phrase->words[i], words[i]);
        }
        
        db->count++;
        db->total_phrases++;
        return 1;
    }
    
    return 0; // Database full
}

// Add collocation to database
int add_collocation_to_database(CollocationDatabase *db, const char *word1, const char *word2) {
    // Check if collocation already exists
    for (size_t i = 0; i < db->count; i++) {
        if (strcmp(db->collocations[i].word1, word1) == 0 && 
            strcmp(db->collocations[i].word2, word2) == 0) {
            db->collocations[i].frequency++;
            db->total_collocations++;
            return 1;
        }
    }
    
    // Add new collocation if database not full
    if (db->count < db->max_size) {
        Collocation *colloc = &db->collocations[db->count];
        strcpy(colloc->word1, word1);
        strcpy(colloc->word2, word2);
        colloc->frequency = 1;
        colloc->association_strength = 0.0L;
        
        db->count++;
        db->total_collocations++;
        return 1;
    }
    
    return 0; // Database full
}

// Process text file and extract phrases and collocations
int process_text_file(PhraseDatabase *phrase_db, CollocationDatabase *colloc_db, const char *filename) {
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
    
    // Extract sliding window phrases
    int pos = 0;
    int phrases_added = 0;
    int collocations_added = 0;
    
    while (pos < read_size) {
        char words[MAX_PHRASE_LENGTH + 2][MAX_WORD_LENGTH]; // +2 for context
        int word_count = extract_words_from_text(text, pos, words, MAX_PHRASE_LENGTH + 2);
        
        if (word_count >= 2) {
            // Extract phrases of different lengths
            for (int length = 2; length <= MAX_PHRASE_LENGTH && length <= word_count; length++) {
                char phrase_words[MAX_PHRASE_LENGTH][MAX_WORD_LENGTH];
                for (int i = 0; i < length; i++) {
                    strcpy(phrase_words[i], words[i]);
                }
                
                if (add_phrase_to_database(phrase_db, phrase_words, length)) {
                    phrases_added++;
                }
            }
            
            // Extract collocations (adjacent word pairs)
            for (int i = 0; i < word_count - 1; i++) {
                if (add_collocation_to_database(colloc_db, words[i], words[i + 1])) {
                    collocations_added++;
                }
            }
        }
        
        // Move to next word
        while (pos < read_size && !is_word_char(text[pos])) {
            pos++;
        }
        while (pos < read_size && is_word_char(text[pos])) {
            pos++;
        }
    }
    
    free(text);
    printf("Added %d phrases and %d collocations from %s\n", phrases_added, collocations_added, filename);
    return phrases_added + collocations_added;
}

// Sort phrases by frequency (descending)
void sort_phrases_by_frequency(PhraseDatabase *db) {
    for (size_t i = 0; i < db->count - 1; i++) {
        for (size_t j = i + 1; j < db->count; j++) {
            if (db->phrases[j].frequency > db->phrases[i].frequency) {
                Phrase temp = db->phrases[i];
                db->phrases[i] = db->phrases[j];
                db->phrases[j] = temp;
            }
        }
    }
}

// Sort collocations by frequency (descending)
void sort_collocations_by_frequency(CollocationDatabase *db) {
    for (size_t i = 0; i < db->count - 1; i++) {
        for (size_t j = i + 1; j < db->count; j++) {
            if (db->collocations[j].frequency > db->collocations[i].frequency) {
                Collocation temp = db->collocations[i];
                db->collocations[i] = db->collocations[j];
                db->collocations[j] = temp;
            }
        }
    }
}

// Create phrase vectors from vocabulary
void create_phrase_vectors(PhraseDatabase *phrase_db, Vocabulary *vocab, int vector_length) {
    printf("Creating phrase vectors (length: %d)...\n", vector_length);
    
    for (size_t i = 0; i < phrase_db->count; i++) {
        Phrase *phrase = &phrase_db->phrases[i];
        
        // Allocate vector
        phrase->vector = malloc(vector_length * sizeof(long double));
        phrase->vector_length = vector_length;
        
        if (!phrase->vector) {
            printf("Memory allocation failed for phrase\n");
            continue;
        }
        
        // Initialize vector with zeros
        for (int j = 0; j < vector_length; j++) {
            phrase->vector[j] = 0.0L;
        }
        
        // Add word vectors
        int valid_words = 0;
        for (int j = 0; j < phrase->word_count; j++) {
            WordEntry *word = find_word(vocab, phrase->words[j]);
            if (word && word->vector) {
                for (int k = 0; k < vector_length && k < word->vector_length; k++) {
                    phrase->vector[k] += word->vector[k];
                }
                valid_words++;
            }
        }
        
        // Average the vectors
        if (valid_words > 0) {
            for (int j = 0; j < vector_length; j++) {
                phrase->vector[j] /= valid_words;
            }
        }
        
        // Add frequency information
        long double freq_norm = (long double)phrase->frequency / phrase_db->total_phrases;
        for (int j = 0; j < vector_length; j++) {
            phrase->vector[j] += freq_norm * 0.1L;
        }
    }
}

// Calculate collocation association strengths
void calculate_collocation_strengths(CollocationDatabase *colloc_db, Vocabulary *vocab) {
    printf("Calculating collocation association strengths...\n");
    
    for (size_t i = 0; i < colloc_db->count; i++) {
        Collocation *colloc = &colloc_db->collocations[i];
        
        WordEntry *word1 = find_word(vocab, colloc->word1);
        WordEntry *word2 = find_word(vocab, colloc->word2);
        
        if (word1 && word2) {
            // Simple association strength based on frequency and co-occurrence
            long double prob1 = (long double)word1->frequency / vocab->total_words;
            long double prob2 = (long double)word2->frequency / vocab->total_words;
            long double joint_prob = (long double)colloc->frequency / colloc_db->total_collocations;
            
            // Pointwise Mutual Information approximation
            if (prob1 > 0 && prob2 > 0 && joint_prob > 0) {
                colloc->association_strength = logl(joint_prob / (prob1 * prob2));
            } else {
                colloc->association_strength = 0.0L;
            }
        }
    }
}

// Save phrase database to file
int save_phrase_database(PhraseDatabase *db, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Error: Cannot create phrase file %s\n", filename);
        return 0;
    }
    
    fprintf(file, "Phrase Database\n");
    fprintf(file, "Count: %zu\n", db->count);
    fprintf(file, "Total phrases: %d\n", db->total_phrases);
    fprintf(file, "Vector length: %d\n", db->phrases[0].vector_length);
    fprintf(file, "\n");
    
    for (size_t i = 0; i < db->count; i++) {
        Phrase *phrase = &db->phrases[i];
        fprintf(file, "%d ", phrase->word_count);
        
        for (int j = 0; j < phrase->word_count; j++) {
            fprintf(file, "%s ", phrase->words[j]);
        }
        
        fprintf(file, "%d ", phrase->frequency);
        
        // Save vector values
        for (int j = 0; j < phrase->vector_length; j++) {
            fprintf(file, "%.6Lf ", phrase->vector[j]);
        }
        fprintf(file, "\n");
    }
    
    fclose(file);
    printf("Phrase database saved to %s\n", filename);
    return 1;
}

// Save collocation database to file
int save_collocation_database(CollocationDatabase *db, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Error: Cannot create collocation file %s\n", filename);
        return 0;
    }
    
    fprintf(file, "Collocation Database\n");
    fprintf(file, "Count: %zu\n", db->count);
    fprintf(file, "Total collocations: %d\n", db->total_collocations);
    fprintf(file, "\n");
    
    for (size_t i = 0; i < db->count; i++) {
        Collocation *colloc = &db->collocations[i];
        fprintf(file, "%s %s %d %.6Lf\n", colloc->word1, colloc->word2, 
                colloc->frequency, colloc->association_strength);
    }
    
    fclose(file);
    printf("Collocation database saved to %s\n", filename);
    return 1;
}

// Print phrase database statistics
void print_phrase_statistics(PhraseDatabase *phrase_db, CollocationDatabase *colloc_db) {
    printf("\n=== Phrase Database Statistics ===\n");
    printf("Total phrases: %zu\n", phrase_db->count);
    printf("Total phrase instances: %d\n", phrase_db->total_phrases);
    printf("Vector length: %d\n", phrase_db->phrases[0].vector_length);
    
    if (phrase_db->count > 0) {
        printf("Most frequent phrases:\n");
        int show_count = phrase_db->count < 15 ? phrase_db->count : 15;
        for (int i = 0; i < show_count; i++) {
            Phrase *phrase = &phrase_db->phrases[i];
            printf("  %d. ", i + 1);
            for (int j = 0; j < phrase->word_count; j++) {
                printf("%s ", phrase->words[j]);
            }
            printf("(%d times)\n", phrase->frequency);
        }
    }
    
    printf("\n=== Collocation Database Statistics ===\n");
    printf("Total collocations: %zu\n", colloc_db->count);
    printf("Total collocation instances: %d\n", colloc_db->total_collocations);
    
    if (colloc_db->count > 0) {
        printf("Strongest collocations:\n");
        int show_count = colloc_db->count < 10 ? colloc_db->count : 10;
        for (int i = 0; i < show_count; i++) {
            Collocation *colloc = &colloc_db->collocations[i];
            printf("  %d. %s %s (%d times, strength: %.6Lf)\n", 
                   i + 1, colloc->word1, colloc->word2, 
                   colloc->frequency, colloc->association_strength);
        }
    }
    printf("\n");
}

// Cleanup databases
void cleanup_phrase_database(PhraseDatabase *db) {
    if (db) {
        for (size_t i = 0; i < db->count; i++) {
            free(db->phrases[i].vector);
        }
        free(db->phrases);
        free(db);
    }
}

void cleanup_collocation_database(CollocationDatabase *db) {
    if (db) {
        free(db->collocations);
        free(db);
    }
}

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
    printf("=== Stage 3: Phrase Extraction and Collocation Analysis ===\n\n");
    
    const char *input_file = "training_data/raw_texts/Frankenstein.txt";
    const char *vocab_file = "stage2_vocabulary.txt";
    const char *phrase_file = "stage3_phrases.txt";
    const char *colloc_file = "stage3_collocations.txt";
    int vector_length = 64;
    
    if (argc > 1) input_file = argv[1];
    if (argc > 2) vocab_file = argv[2];
    if (argc > 3) phrase_file = argv[3];
    if (argc > 4) colloc_file = argv[4];
    if (argc > 5) vector_length = atoi(argv[5]);
    
    printf("Configuration:\n");
    printf("  Input file: %s\n", input_file);
    printf("  Vocabulary file: %s\n", vocab_file);
    printf("  Phrase file: %s\n", phrase_file);
    printf("  Collocation file: %s\n", colloc_file);
    printf("  Vector length: %d\n", vector_length);
    printf("\n");
    
    // Load vocabulary
    printf("Loading vocabulary...\n");
    Vocabulary *vocab = load_vocabulary(vocab_file);
    if (!vocab) {
        printf("Failed to load vocabulary\n");
        return 1;
    }
    printf("✓ Vocabulary loaded: %zu words\n", vocab->count);
    
    // Initialize databases
    PhraseDatabase *phrase_db = init_phrase_database(MAX_PHRASES);
    CollocationDatabase *colloc_db = init_collocation_database(MAX_PHRASES);
    
    if (!phrase_db || !colloc_db) {
        printf("Failed to initialize databases\n");
        cleanup_vocabulary(vocab);
        return 1;
    }
    
    // Process text file
    printf("\nExtracting phrases and collocations...\n");
    int items_added = process_text_file(phrase_db, colloc_db, input_file);
    
    if (items_added > 0) {
        // Sort by frequency
        sort_phrases_by_frequency(phrase_db);
        sort_collocations_by_frequency(colloc_db);
        
        // Create phrase vectors
        create_phrase_vectors(phrase_db, vocab, vector_length);
        
        // Calculate collocation strengths
        calculate_collocation_strengths(colloc_db, vocab);
        
        // Print statistics
        print_phrase_statistics(phrase_db, colloc_db);
        
        // Save databases
        save_phrase_database(phrase_db, phrase_file);
        save_collocation_database(colloc_db, colloc_file);
        
    } else {
        printf("No phrases or collocations were added\n");
    }
    
    // Cleanup
    cleanup_phrase_database(phrase_db);
    cleanup_collocation_database(colloc_db);
    cleanup_vocabulary(vocab);
    
    printf("\n=== Stage 3 Phrase Extraction Completed ===\n");
    printf("Ready for Stage 3: Phrase Grouping Training\n");
    
    return 0;
}
