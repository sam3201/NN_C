#ifndef LLM_NEAT_H
#define LLM_NEAT_H

#include "NN.h"
#include <stdlib.h>

#define MAX_SEQ_LENGTH 512

// Neural network type for NEAT
typedef NN_t NEAT_NN_t;

// Attention mechanism structures
typedef struct NEAT_MultiHeadAttention {
    size_t model_dim;
    size_t num_heads;
    size_t head_dim;
    NEAT_NN_t *Q_proj;
    NEAT_NN_t *K_proj;
    NEAT_NN_t *V_proj;
    NEAT_NN_t *O_proj;
    long double *attention_scores;
    long double *attention_probs;
    long double *attention_output;
} NEAT_MultiHeadAttention;

// Feed-forward network structure
typedef struct NEAT_FeedForward {
    size_t input_dim;
    size_t hidden_dim;
    NEAT_NN_t *network;
} NEAT_FeedForward;

// Layer normalization structure
typedef struct NEAT_LayerNorm {
    size_t dim;
    long double epsilon;
    NEAT_NN_t *norm_network;
    long double *mean;
    long double *var;
    long double *normalized;
} NEAT_LayerNorm;

// NEAT history tracking
typedef struct {
    int innovation_number;
    int generation;
    int species_id;
} NEAT_History;

// Main transformer structure with NEAT components
typedef struct NEAT_Transformer {
    size_t input_dim;
    NEAT_MultiHeadAttention *self_attention;
    NEAT_LayerNorm *norm1;
    NEAT_FeedForward *feed_forward;
    NEAT_LayerNorm *norm2;
    NEAT_History neat_history;
    long double fitness;
} NEAT_Transformer;

// Function declarations
NEAT_MultiHeadAttention* create_attention(size_t model_dim, size_t num_heads);
NEAT_FeedForward* create_feed_forward(size_t input_dim, size_t hidden_dim);
NEAT_LayerNorm* create_layer_norm(size_t dim, long double epsilon);
NEAT_Transformer* create_neat_transformer(size_t input_dim, size_t num_heads);

// Forward and backward pass
long double* llm_neat_forward(NEAT_Transformer *transformer, long double *input);
void llm_neat_backprop(NEAT_Transformer *transformer, long double *input, long double *grad_output, long double *grad_input);
void llm_neat_update(NEAT_Transformer *transformer, long double learning_rate);

// NEAT-specific operations
void neat_mutate(NEAT_Transformer *transformer, long double mutation_rate);
NEAT_Transformer* neat_crossover(NEAT_Transformer *parent1, NEAT_Transformer *parent2);
void calculate_fitness(NEAT_Transformer *transformer, long double *input, long double *target);

// Memory management
void transformer_free(NEAT_Transformer *transformer);

// Neural network operations
#define NEAT_NN_init NN_init
#define NEAT_NN_forward NN_forward
#define NEAT_NN_backprop NN_backprop
#define NEAT_NN_update NN_update
#define NEAT_NN_mutate NN_mutate
#define NEAT_NN_copy NN_copy
#define NEAT_NN_destroy NN_destroy

#endif // LLM_NEAT_H
