#ifndef FORERUNNER_H
#define FORERUNNER_H

#include "../utils/NN/TRANSFORMER.h"
#include "../utils/NN/NN.h"
#include <stdlib.h>
#include <stdbool.h>

#define NONE 0
#define TEXT 1
#define IMAGE 2
#define VIDEO 3
#define CONTENTS {NONE, TEXT, IMAGE, VIDEO}
#define NUM_CONTENTS sizeof(CONTENTS) / sizeof(CONTENTS[0])

// ForeRunner Model structure
typedef struct {
    Transformer_t* transformer;  // Transformer model for feature extraction
    unsigned int num_contexts;   // Number of possible contexts
} ForeRunner_t;

// Initialize the ForeRunner model
ForeRunner_t* ForeRunner_init(size_t input_dim, size_t num_heads, unsigned int num_contexts);

// Compute context from input data
unsigned int ForeRunner_forward(ForeRunner_t* model, long double** input_sequence, unsigned int seq_length);
    
// Backpropagation function
void ForeRunner_backprop(ForeRunner_t* model, long double** input_sequence, unsigned int seq_length, long double* grad_loss);

// Destroy the ForeRunner model
void ForeRunner_destroy(ForeRunner_t* model);

#endif // FORERUNNER_H
