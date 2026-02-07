#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "../NN/NN.h"
#include <stddef.h>

#define MAX_VOCAB_SIZE 10000
#define MAX_TOKEN_LENGTH 50

typedef struct {
  char **vocabulary;
  size_t vocab_size;
  size_t max_sequence_length;
} Tokenizer;

// Function declarations
Tokenizer *create_tokenizer(const char *corpus_path);
void tokenizer_fit(Tokenizer *tokenizer, const char *text);
long double *tokenizer_encode(Tokenizer *tokenizer, const char *text);
char *tokenizer_decode(Tokenizer *tokenizer, const long double *encoded);
void free_tokenizer(Tokenizer *tokenizer);

#endif // TOKENIZER_H
