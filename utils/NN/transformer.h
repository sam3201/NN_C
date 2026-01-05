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

  // Intermediate caches for backprop
  long double *attention_input;  // input to MHA
  long double *attention_output; // output of MHA
  long double *norm1_input;      // input to norm1 (after residual)
  long double *norm1_output;     // output of norm1
  long double *ff_input;         // input to feedforward
  long double *ff_output;        // output of feedforward
  long double *norm2_input;      // input to norm2 (after residual)
  long double *norm2_output;     // output of norm2
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
long double *transformer_forward(TransformerLayer *layer, long double *input) {
  if (!layer || !input)
    return NULL;

  layer->attention_input = malloc(layer->model_dim * sizeof(long double));
  memcpy(layer->attention_input, input, layer->model_dim * sizeof(long double));

  layer->attention_output =
      transformer_mha_forward(layer->attention, input, layer->seq_length);
  if (!layer->attention_output)
    return NULL;

  // Residual1
  layer->norm1_input = malloc(layer->model_dim * sizeof(long double));
  for (size_t i = 0; i < layer->model_dim; i++)
    layer->norm1_input[i] = input[i] + layer->attention_output[i];

  free(layer->attention_output);
  layer->norm1_output =
      transformer_norm_forward(layer->norm1, layer->norm1_input);
  if (!layer->norm1_output)
    return NULL;

  // FeedForward
  layer->ff_input = malloc(layer->model_dim * sizeof(long double));
  memcpy(layer->ff_input, layer->norm1_output,
         layer->model_dim * sizeof(long double));
  layer->ff_output = NN_forward(layer->feed_forward->network, layer->ff_input);
  if (!layer->ff_output)
    return NULL;

  // Residual2
  layer->norm2_input = malloc(layer->model_dim * sizeof(long double));
  for (size_t i = 0; i < layer->model_dim; i++)
    layer->norm2_input[i] = layer->norm1_output[i] + layer->ff_output[i];

  free(layer->ff_output);
  layer->norm2_output =
      transformer_norm_forward(layer->norm2, layer->norm2_input);
  if (!layer->norm2_output)
    return NULL;

  return layer->norm2_output;
}

// Backpropagation functions
void transformer_mha_backprop(MultiHeadAttention *mha, long double *input);

void transformer_norm_backprop(LayerNorm *ln, long double *input);

void TRANSFORMER_backprop(Transformer_t *transformer,
                          long double **input_sequence, size_t seq_length,
                          long double *grad_loss);

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
