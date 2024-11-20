#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <stdlib.h>
#include "NN.h"

// Multi-head attention structures
typedef struct {
    size_t model_dim;
    size_t num_heads;
    size_t head_dim;
    NN_t *Q_proj;
    NN_t *K_proj;
    NN_t *V_proj;
    NN_t *O_proj;
} MultiHeadAttention;

// Layer normalization structure
typedef struct {
    size_t dim;
    long double epsilon;
    NN_t *norm_network;
} LayerNorm;

// Feed-forward network structure
typedef struct {
    size_t input_dim;
    size_t hidden_dim;
    NN_t *network;
} FeedForward;

// Complete transformer layer structure
typedef struct {
    size_t model_dim;
    size_t seq_length;
    MultiHeadAttention *attention;
    LayerNorm *norm1;
    FeedForward *feed_forward;
    LayerNorm *norm2;
} TransformerLayer;

// Creation functions
MultiHeadAttention* create_attention(size_t model_dim, size_t num_heads);
FeedForward* create_feed_forward(size_t input_dim, size_t hidden_dim);
LayerNorm* create_layer_norm(size_t dim, long double epsilon);
TransformerLayer* create_transformer_layer(size_t model_dim, size_t num_heads, size_t ff_dim);

// Forward pass functions
long double *transformer_mha_forward(MultiHeadAttention *mha, long double *input, size_t seq_length);
long double *transformer_norm_forward(LayerNorm *ln, long double *input);
long double *transformer_forward(TransformerLayer *layer, long double *input);

// Backpropagation functions
void transformer_mha_backprop(MultiHeadAttention *mha, long double *input, long double *grad_output, long double *grad_input);
void transformer_norm_backprop(LayerNorm *ln, long double *input, long double *grad_output, long double *grad_input);
void transformer_backprop(TransformerLayer *layer, long double *input, long double *grad_output, long double *grad_input);

// Memory management functions
void free_attention(MultiHeadAttention *mha);
void free_feed_forward(FeedForward *ff);
void free_layer_norm(LayerNorm *ln);
void free_transformer_layer(TransformerLayer *layer);

#endif // TRANSFORMER_H
