#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "NN.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Multi-head attention
typedef struct {
  size_t model_dim;
  size_t num_heads;
  size_t head_dim;

  NN_t *Q_proj;
  NN_t *K_proj;
  NN_t *V_proj;
  NN_t *O_proj;

  // Cached for backprop
  long double *Q_cache;
  long double *K_cache;
  long double *V_cache;
  long double *scores_cache;
  size_t seq_length;
} MultiHeadAttention;

// Layer normalization
typedef struct {
  size_t dim;
  long double epsilon;
  NN_t *norm_network;

  long double *input_cache; // cached input for backprop
} LayerNorm;

// Feed-forward network
typedef struct {
  size_t input_dim;
  size_t hidden_dim;
  NN_t *network;

  long double *input_cache; // cached input for backprop
} FeedForward;

// Transformer layer
typedef struct {
  size_t model_dim;
  size_t seq_length;

  MultiHeadAttention *attention;
  LayerNorm *norm1;
  FeedForward *feed_forward;
  LayerNorm *norm2;

  // Cached intermediate states
  long double *attention_input;
  long double *norm1_input;
  long double *ff_input;
  long double *norm2_input;
} TransformerLayer;

// Complete Transformer
typedef struct {
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
// ----------------------
// Init + Train (MISSING IN YOUR FILE)
// ----------------------

static void free_seq(long double **seq, size_t T) {
  if (!seq)
    return;
  for (size_t t = 0; t < T; t++)
    free(seq[t]);
  free(seq);
}

Transformer_t *TRANSFORMER_init(size_t model_dim, size_t num_heads,
                                size_t num_layers) {
  if (model_dim == 0 || num_heads == 0 || num_layers == 0)
    return NULL;
  if (model_dim % num_heads != 0)
    return NULL; // head_dim must be integer

  Transformer_t *t = (Transformer_t *)calloc(1, sizeof(Transformer_t));
  if (!t)
    return NULL;

  t->model_dim = model_dim;
  t->num_heads = num_heads;
  t->num_layers = num_layers;

  t->layers =
      (TransformerLayer **)calloc(num_layers, sizeof(TransformerLayer *));
  if (!t->layers) {
    free(t);
    return NULL;
  }

  size_t ff_dim = model_dim * 4; // common default
  for (size_t i = 0; i < num_layers; i++) {
    t->layers[i] = create_transformer_layer(model_dim, num_heads, ff_dim);
    if (!t->layers[i]) {
      for (size_t j = 0; j < i; j++)
        free_transformer_layer(t->layers[j]);
      free(t->layers);
      free(t);
      return NULL;
    }
  }

  return t;
}

void TRANSFORMER_train(Transformer_t *transformer, long double **input_sequence,
                       size_t seq_length, long double *target);
// Forward pass
long double *transformer_norm_forward(LayerNorm *ln, long double *input);
long double **transformer_layer_forward(TransformerLayer *layer,
                                        long double **input, size_t seq_length);
long double **TRANSFORMER_forward(Transformer_t *transformer,
                                  long double **input_sequence,
                                  size_t seq_length);

// Backpropagation
void transformer_mha_backprop(MultiHeadAttention *mha,
                              long double *grad_output);
void transformer_layernorm_backprop(LayerNorm *ln, long double *grad_output);
void transformer_feedforward_backprop(FeedForward *ff,
                                      long double *grad_output);
long double *transformer_layer_backprop(TransformerLayer *layer,
                                        long double *grad_output);
long double **TRANSFORMER_backprop(Transformer_t *transformer,
                                   long double **grad_output,
                                   size_t seq_length);

// Training
void TRANSFORMER_train(Transformer_t *transformer, long double **input_sequence,
                       size_t seq_length, long double *target);

// Free memory
void free_attention(MultiHeadAttention *mha);
void free_feed_forward(FeedForward *ff);
void free_layer_norm(LayerNorm *ln);
void free_transformer_layer(TransformerLayer *layer);
void TRANSFORMER_destroy(Transformer_t *transformer);

// Serialization
int TRANSFORMER_save(Transformer_t *t, FILE *file);
Transformer_t *TRANSFORMER_load(FILE *file);

#endif // TRANSFORMER_H
