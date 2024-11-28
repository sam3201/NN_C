#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Helper function to check if word exists in vocabulary
int word_exists(Tokenizer* tokenizer, const char* word) {
    for (size_t i = 0; i < tokenizer->vocab_size; i++) {
        if (strcmp(tokenizer->vocabulary[i], word) == 0) {
            return i;
        }
    }
    return -1;
}

// Create and initialize tokenizer
Tokenizer* create_tokenizer(const char* corpus_path) {
    Tokenizer* tokenizer = (Tokenizer*)malloc(sizeof(Tokenizer));
    if (!tokenizer) return NULL;

    tokenizer->vocabulary = (char**)malloc(MAX_VOCAB_SIZE * sizeof(char*));
    tokenizer->vocab_size = 0;
    tokenizer->max_sequence_length = 0;

    // Read corpus and build vocabulary
    FILE* file = fopen(corpus_path, "r");
    if (!file) {
        fprintf(stderr, "Failed to open corpus file: %s\n", corpus_path);
        free(tokenizer);
        return NULL;
    }

    char line[1024];
    while (fgets(line, sizeof(line), file)) {
        tokenizer_fit(tokenizer, line);
    }

    fclose(file);
    return tokenizer;
}

// Fit tokenizer on text
void tokenizer_fit(Tokenizer* tokenizer, const char* text) {
    char* text_copy = strdup(text);
    char* word = strtok(text_copy, " \t\n\r");
    
    size_t sequence_length = 0;
    while (word && tokenizer->vocab_size < MAX_VOCAB_SIZE) {
        sequence_length++;
        
        // Convert to lowercase
        for (char* p = word; *p; p++) {
            *p = tolower(*p);
        }

        if (word_exists(tokenizer, word) == -1) {
            tokenizer->vocabulary[tokenizer->vocab_size] = strdup(word);
            tokenizer->vocab_size++;
        }
        
        word = strtok(NULL, " \t\n\r");
    }

    if (sequence_length > tokenizer->max_sequence_length) {
        tokenizer->max_sequence_length = sequence_length;
    }

    free(text_copy);
}

// Encode text to vector
long double* tokenizer_encode(Tokenizer* tokenizer, const char* text) {
    long double* encoded = (long double*)calloc(tokenizer->max_sequence_length, sizeof(long double));
    
    char* text_copy = strdup(text);
    char* word = strtok(text_copy, " \t\n\r");
    
    size_t pos = 0;
    while (word && pos < tokenizer->max_sequence_length) {
        // Convert to lowercase
        for (char* p = word; *p; p++) {
            *p = tolower(*p);
        }

        int index = word_exists(tokenizer, word);
        if (index != -1) {
            encoded[pos] = (long double)index;
        }
        
        pos++;
        word = strtok(NULL, " \t\n\r");
    }

    free(text_copy);
    return encoded;
}

// Decode vector to text
char* tokenizer_decode(Tokenizer* tokenizer, const long double* encoded) {
    char decoded[4096] = {0}; // Adjust size as needed
    
    for (size_t i = 0; i < tokenizer->max_sequence_length; i++) {
        int index = (int)encoded[i];
        if (index >= 0 && index < tokenizer->vocab_size) {
            strcat(decoded, tokenizer->vocabulary[index]);
            strcat(decoded, " ");
        }
    }

    return strdup(decoded);
}

// Free tokenizer
void free_tokenizer(Tokenizer* tokenizer) {
    if (!tokenizer) return;

    for (size_t i = 0; i < tokenizer->vocab_size; i++) {
        free(tokenizer->vocabulary[i]);
    }
    free(tokenizer->vocabulary);
    free(tokenizer);
}
