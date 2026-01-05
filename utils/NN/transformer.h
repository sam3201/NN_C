#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "NN.h"
#include <stdio.h>
#include <stdlib.h>

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

// Complete transformer structure
typedef struct Transformer_t {
  size_t model_dim;
  size_t num_heads;
  size_t num_layers;
  TransformerLayer **layers;
} Transformer_t;

// Creation functions
MultiHeadAttention *create_attention(size_t model_dim, size_t num_heads);
FeedForward *create_feed_forward(size_t input_dim, size_t hidden_dim);
LayerNorm *create_layer_norm(size_t dim, long double epsilon);
TransformerLayer *create_transformer_layer(size_t model_dim, size_t num_heads,
                                           size_t ff_dim);
Transformer_t *TRANSFORMER_init(size_t input_dim, size_t num_heads);

// Forward pass functions
long double *transformer_mha_forward(MultiHeadAttention *mha,
                                     long double *input, size_t seq_length);
long double *transformer_norm_forward(LayerNorm *ln, long double *input);
long double *transformer_forward(TransformerLayer *layer, long double *input);
long double *TRANSFORMER_forward(Transformer_t *transformer,
                                 long double **input_sequence,
                                 size_t seq_length);

// Backpropagation functions
void transformer_mha_backprop(MultiHeadAttention *mha, long double *input,
                              long double *grad_output,
                              long double *grad_input);
void transformer_norm_backprop(LayerNorm *ln, long double *input,
                               long double *grad_output,
                               long double *grad_input);

void TRANSFORMER_backprop(TransformerLayer *layer, long double *input,
                          long double *grad_output, long double *grad_input);

void TRANSFORMER_train(Transformer_t *transformer, long double **input_sequence,
                       size_t seq_length, long double *target);

// Memory management functions
void free_attention(MultiHeadAttention *mha);
void free_feed_forward(FeedForward *ff);
void free_layer_norm(LayerNorm *ln);
void free_transformer_layer(TransformerLayer *layer);
void TRANSFORMER_destroy(Transformer_t *transformer);

// Serialization functions
int TRANSFORMER_save(Transformer_t *transformer, FILE *file);
Transformer_t *TRANSFORMER_load(FILE *file);

#endif // TRANSFORMER_H
