#ifndef LLM_NEAT_H
#define LLM_NEAT_H

#include <stdlib.h>
#include "NN.h"

// Forward declarations
typedef struct NEAT_NN NEAT_NN_t;
typedef struct NEAT_MultiHeadAttention NEAT_MultiHeadAttention;
typedef struct NEAT_FeedForward NEAT_FeedForward;
typedef struct NEAT_LayerNorm NEAT_LayerNorm;
typedef struct NEAT_Transformer NEAT_Transformer;

// Structures
struct NEAT_NN {
    size_t input_dim;
    size_t output_dim;
    size_t num_layers;
    long double* weights;
    long double* biases;
    ActivationFunctionType* activation_functions;
    ActivationDerivativeType* activation_derivatives;
    RegularizationType regularization_type;
    Optimizer* optimizer;
    long double learning_rate;
};

struct NEAT_MultiHeadAttention {
    size_t model_dim;
    size_t num_heads;
    size_t head_dim;
    NEAT_NN_t* Q_proj;
    NEAT_NN_t* K_proj;
    NEAT_NN_t* V_proj;
    NEAT_NN_t* O_proj;
};

struct NEAT_FeedForward {
    size_t input_dim;
    size_t hidden_dim;
    NEAT_NN_t* network;
};

struct NEAT_LayerNorm {
    size_t dim;
    long double epsilon;
    long double* gamma;
    long double* beta;
};

struct NEAT_Transformer {
    size_t input_dim;
    size_t model_dim;
    size_t num_heads;
    NEAT_MultiHeadAttention* self_attention;
    NEAT_FeedForward* feed_forward;
    NEAT_LayerNorm* layer_norm1;
    NEAT_LayerNorm* layer_norm2;
};

// Creation and initialization functions
NEAT_NN_t* create_basic_network(size_t input_dim, size_t output_dim);
NEAT_MultiHeadAttention* create_multi_head_attention(size_t model_dim, size_t num_heads);
NEAT_FeedForward* create_feed_forward(size_t input_dim, size_t hidden_dim);
NEAT_LayerNorm* create_layer_norm(size_t dim, long double epsilon);
NEAT_Transformer* create_neat_transformer(size_t input_dim, size_t model_dim, size_t num_heads);

// Core functions for neural evolution
NEAT_Transformer* neat_crossover(NEAT_Transformer* parent1, NEAT_Transformer* parent2);
void neat_mutate(NEAT_Transformer* transformer, long double mutation_rate);
void optimize_architecture(NEAT_Transformer* transformer, long double* input, long double* target);
long double calculate_fitness(NEAT_Transformer* transformer, long double* input, long double* target);

// Memory management functions
void NN_free(NEAT_NN_t* nn);
void free_multi_head_attention(NEAT_MultiHeadAttention* mha);
void free_feed_forward(NEAT_FeedForward* ff);
void free_layer_norm(NEAT_LayerNorm* ln);
void free_neat_transformer(NEAT_Transformer* transformer);

// Forward pass function
long double* llm_neat_forward(NEAT_Transformer* transformer, long double* input);

#endif // LLM_NEAT_H
