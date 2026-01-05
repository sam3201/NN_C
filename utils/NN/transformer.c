#include "TRANSFORMER.h"
#include "NN.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Creation functions
MultiHeadAttention *create_attention(size_t model_dim, size_t num_heads) {
  MultiHeadAttention *mha = malloc(sizeof(MultiHeadAttention));
  if (!mha) {
    return NULL;
  }

  mha->model_dim = model_dim;
  mha->num_heads = num_heads;
  mha->head_dim = model_dim / num_heads;

  // Create projection networks
  size_t layers[] = {model_dim, mha->head_dim, 0};
  ActivationFunctionType activations[] = {RELU, RELU};
  ActivationDerivativeType derivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE};

  mha->Q_proj = NN_init(layers, activations, derivatives, MSE, MSE_DERIVATIVE,
                        L1, SGD, 0.01L);
  mha->K_proj = NN_init(layers, activations, derivatives, MSE, MSE_DERIVATIVE,
                        L1, SGD, 0.01L);
  mha->V_proj = NN_init(layers, activations, derivatives, MSE, MSE_DERIVATIVE,
                        L1, SGD, 0.01L);
  mha->O_proj = NN_init(layers, activations, derivatives, MSE, MSE_DERIVATIVE,
                        L1, SGD, 0.01L);

  if (!mha->Q_proj || !mha->K_proj || !mha->V_proj || !mha->O_proj) {
    free_attention(mha);
    return NULL;
  }

  return mha;
}

FeedForward *create_feed_forward(size_t input_dim, size_t hidden_dim) {
  FeedForward *ff = malloc(sizeof(FeedForward));
  if (!ff) {
    return NULL;
  }

  ff->input_dim = input_dim;
  ff->hidden_dim = hidden_dim;

  size_t layers[] = {input_dim, hidden_dim, input_dim, 0};
  ActivationFunctionType activations[] = {RELU, RELU, RELU};
  ActivationDerivativeType derivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE,
                                            RELU_DERIVATIVE};

  ff->network = NN_init(layers, activations, derivatives, MSE, MSE_DERIVATIVE,
                        L1, SGD, 0.01L);
  if (!ff->network) {
    free(ff);
    return NULL;
  }

  return ff;
}

LayerNorm *create_layer_norm(size_t dim, long double epsilon) {
  LayerNorm *ln = malloc(sizeof(LayerNorm));
  if (!ln) {
    return NULL;
  }

  ln->dim = dim;
  ln->epsilon = epsilon;

  size_t layers[] = {dim, dim, 0};
  ActivationFunctionType activations[] = {RELU, RELU};
  ActivationDerivativeType derivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE};

  ln->norm_network = NN_init(layers, activations, derivatives, MSE,
                             MSE_DERIVATIVE, L1, SGD, 0.01L);
  if (!ln->norm_network) {
    free(ln);
    return NULL;
  }

  return ln;
}

TransformerLayer *create_transformer_layer(size_t model_dim, size_t num_heads,
                                           size_t ff_dim) {
  TransformerLayer *layer = malloc(sizeof(TransformerLayer));
  if (!layer) {
    return NULL;
  }

  layer->model_dim = model_dim;
  layer->seq_length = 1; // Default to 1, can be changed during forward pass

  // Create multi-head attention
  layer->attention = create_attention(model_dim, num_heads);
  if (!layer->attention) {
    free(layer);
    return NULL;
  }

  // Create first layer norm
  layer->norm1 = create_layer_norm(model_dim, 1e-6L);
  if (!layer->norm1) {
    free_attention(layer->attention);
    free(layer);
    return NULL;
  }

  // Create feed-forward network
  layer->feed_forward = create_feed_forward(model_dim, ff_dim);
  if (!layer->feed_forward) {
    free_attention(layer->attention);
    free_layer_norm(layer->norm1);
    free(layer);
    return NULL;
  }

  // Create second layer norm
  layer->norm2 = create_layer_norm(model_dim, 1e-6L);
  if (!layer->norm2) {
    free_attention(layer->attention);
    free_layer_norm(layer->norm1);
    free_feed_forward(layer->feed_forward);
    free(layer);
    return NULL;
  }

  return layer;
}

// Forward pass functions
long double *transformer_mha_forward(MultiHeadAttention *mha,
                                     long double *input, size_t seq_length) {
  if (!mha || !input || seq_length == 0) {
    return NULL;
  }

  // Project input to Q, K, V matrices
  long double *Q = NN_forward(mha->Q_proj, input);
  if (!Q) {
    return NULL;
  }

  long double *K = NN_forward(mha->K_proj, input);
  if (!K) {
    free(Q);
    return NULL;
  }

  long double *V = NN_forward(mha->V_proj, input);
  if (!V) {
    free(Q);
    free(K);
    return NULL;
  }

  // Calculate attention scores
  long double *scores = malloc(seq_length * seq_length * sizeof(long double));
  if (!scores) {
    free(Q);
    free(K);
    free(V);
    return NULL;
  }

  // Calculate attention scores (Q * K^T / sqrt(d_k))
  for (size_t i = 0; i < seq_length; i++) {
    for (size_t j = 0; j < seq_length; j++) {
      scores[i * seq_length + j] = 0;
      for (size_t k = 0; k < mha->head_dim; k++) {
        scores[i * seq_length + j] +=
            Q[i * mha->head_dim + k] * K[j * mha->head_dim + k];
      }
      scores[i * seq_length + j] /= sqrt(mha->head_dim);
    }
  }

  // Apply softmax
  for (size_t i = 0; i < seq_length; i++) {
    // Find max for numerical stability
    long double max_val = scores[i * seq_length];
    for (size_t j = 1; j < seq_length; j++) {
      if (scores[i * seq_length + j] > max_val) {
        max_val = scores[i * seq_length + j];
      }
    }

    // Calculate exp and sum
    long double sum = 0;
    for (size_t j = 0; j < seq_length; j++) {
      scores[i * seq_length + j] = exp(scores[i * seq_length + j] - max_val);
      sum += scores[i * seq_length + j];
    }

    // Normalize
    for (size_t j = 0; j < seq_length; j++) {
      scores[i * seq_length + j] /= sum;
    }
  }

  // Apply attention to values
  long double *attention_output =
      malloc(seq_length * mha->head_dim * sizeof(long double));
  if (!attention_output) {
    free(Q);
    free(K);
    free(V);
    free(scores);
    return NULL;
  }

  // Calculate attention output (scores * V)
  for (size_t i = 0; i < seq_length; i++) {
    for (size_t j = 0; j < mha->head_dim; j++) {
      attention_output[i * mha->head_dim + j] = 0;
      for (size_t k = 0; k < seq_length; k++) {
        attention_output[i * mha->head_dim + j] +=
            scores[i * seq_length + k] * V[k * mha->head_dim + j];
      }
    }
  }

  // Project output
  long double *final_output = NN_forward(mha->O_proj, attention_output);

  // Clean up
  free(Q);
  free(K);
  free(V);
  free(scores);
  free(attention_output);

  return final_output;
}

long double *transformer_norm_forward(LayerNorm *ln, long double *input) {
  if (!ln || !input) {
    return NULL;
  }

  // Calculate mean
  long double mean = 0;
  for (size_t i = 0; i < ln->dim; i++) {
    mean += input[i];
  }
  mean /= ln->dim;

  // Calculate variance
  long double var = 0;
  for (size_t i = 0; i < ln->dim; i++) {
    var += (input[i] - mean) * (input[i] - mean);
  }
  var /= ln->dim;

  // Normalize
  long double *normalized = malloc(ln->dim * sizeof(long double));
  if (!normalized) {
    return NULL;
  }

  for (size_t i = 0; i < ln->dim; i++) {
    normalized[i] = (input[i] - mean) / sqrt(var + ln->epsilon);
  }

  // Apply affine transformation
  long double *output = NN_forward(ln->norm_network, normalized);
  free(normalized);

  return output;
}

long double *transformer_forward(TransformerLayer *layer, long double *input) {
  if (!layer || !input) {
    return NULL;
  }

  // Multi-head attention
  long double *attention_output =
      transformer_mha_forward(layer->attention, input, layer->seq_length);
  if (!attention_output) {
    return NULL;
  }

  // Add & Norm
  long double *residual1 = malloc(layer->model_dim * sizeof(long double));
  if (!residual1) {
    free(attention_output);
    return NULL;
  }

  for (size_t i = 0; i < layer->model_dim; i++) {
    residual1[i] = input[i] + attention_output[i];
  }
  free(attention_output);

  long double *norm1_output = transformer_norm_forward(layer->norm1, residual1);
  free(residual1);
  if (!norm1_output) {
    return NULL;
  }

  // Feed forward
  long double *ff_output =
      NN_forward(layer->feed_forward->network, norm1_output);
  if (!ff_output) {
    free(norm1_output);
    return NULL;
  }

  // Add & Norm
  long double *residual2 = malloc(layer->model_dim * sizeof(long double));
  if (!residual2) {
    free(norm1_output);
    free(ff_output);
    return NULL;
  }

  for (size_t i = 0; i < layer->model_dim; i++) {
    residual2[i] = norm1_output[i] + ff_output[i];
  }
  free(norm1_output);
  free(ff_output);

  long double *output = transformer_norm_forward(layer->norm2, residual2);
  free(residual2);

  return output;
}

// Backpropagation functions
void transformer_mha_backprop(MultiHeadAttention *mha, long double *input) {
  if (!mha || !input)
    return;

  // Update output projection
  NN_backprop(mha->O_proj, input, 0.0L, 0.0L);

  // Update Q, K, V projections
  NN_backprop(mha->Q_proj, input, 0.0L, 0.0L);
  NN_backprop(mha->K_proj, input, 0.0L, 0.0L);
  NN_backprop(mha->V_proj, input, 0.0L, 0.0L);
}

void transformer_norm_backprop(LayerNorm *ln, long double *input) {
  if (!ln || !input)
    return;

  // Only update the internal affine NN of LayerNorm
  NN_backprop(ln->norm_network, input, 0.0L, 0.0L);
}

void TRANSFORMER_backprop(Transformer_t *transformer,
                          long double **input_sequence, size_t seq_length,
                          long double *grad_loss) {
  if (!transformer || !input_sequence || !grad_loss)
    return;

  size_t L = transformer->num_layers;
  size_t D = transformer->model_dim;

  // Forward pass for caching (needed if not done yet)
  long double *output =
      TRANSFORMER_forward(transformer, input_sequence, seq_length);
  if (!output)
    return;

  // Initialize gradient from loss
  long double *grad = malloc(D * sizeof(long double));
  if (!grad) {
    free(output);
    return;
  }
  memcpy(grad, grad_loss, D * sizeof(long double));

  // Backpropagate through layers in reverse
  for (ssize_t i = L - 1; i >= 0; i--) {
    TransformerLayer *layer = transformer->layers[i];
    if (!layer)
      continue;

    // Gradient w.r.t norm2 input (after residual2)
    long double *grad_norm2_input = malloc(D * sizeof(long double));
    for (size_t j = 0; j < D; j++)
      grad_norm2_input[j] = grad[j];

    // LayerNorm2 backprop
    layernorm_backprop(layer->norm2, grad_norm2_input, layer->norm2_input);

    // Gradient through feedforward + residual1
    NN_backprop(layer->feed_forward->network, layer->ff_input, grad_norm2_input,
                0.01L);

    // Gradient w.r.t norm1 input (after residual1)
    long double *grad_residual1 = malloc(D * sizeof(long double));
    for (size_t j = 0; j < D; j++)
      grad_residual1[j] = grad_norm2_input[j];

    // LayerNorm1 backprop
    layernorm_backprop(layer->norm1, grad_residual1, layer->norm1_input);

    // MHA backprop
    transformer_mha_backprop(layer->attention, grad_residual1,
                             layer->attention_input);

    // Gradient for previous layer
    memcpy(grad, grad_residual1, D * sizeof(long double));

    free(grad_norm2_input);
    free(grad_residual1);
  }

  free(grad);
  free(output);
}

void free_feed_forward(FeedForward *ff) {
  if (ff) {
    if (ff->network)
      NN_destroy(ff->network);
    free(ff);
  }
}

void free_layer_norm(LayerNorm *ln) {
  if (ln) {
    if (ln->norm_network)
      NN_destroy(ln->norm_network);
    free(ln);
  }
}

void free_transformer_layer(TransformerLayer *layer) {
  if (layer) {
    if (layer->attention)
      free_attention(layer->attention);
    if (layer->norm1)
      free_layer_norm(layer->norm1);
    if (layer->feed_forward)
      free_feed_forward(layer->feed_forward);
    if (layer->norm2)
      free_layer_norm(layer->norm2);
    free(layer);
  }
}

// Transformer initialization
Transformer_t *TRANSFORMER_init(size_t input_dim, size_t num_heads) {
  Transformer_t *transformer = (Transformer_t *)malloc(sizeof(Transformer_t));
  if (!transformer)
    return NULL;

  transformer->model_dim = input_dim;
  transformer->num_heads = num_heads;
  transformer->num_layers = 2; // Default to 2 layers

  // Allocate layers
  transformer->layers = (TransformerLayer **)malloc(transformer->num_layers *
                                                    sizeof(TransformerLayer *));
  if (!transformer->layers) {
    free(transformer);
    return NULL;
  }

  // Create transformer layers
  size_t ff_dim =
      input_dim * 4; // Feed-forward dimension is typically 4x model_dim
  for (size_t i = 0; i < transformer->num_layers; i++) {
    transformer->layers[i] =
        create_transformer_layer(input_dim, num_heads, ff_dim);
    if (!transformer->layers[i]) {
      // Cleanup on error
      for (size_t j = 0; j < i; j++) {
        free_transformer_layer(transformer->layers[j]);
      }
      free(transformer->layers);
      free(transformer);
      return NULL;
    }
  }

  return transformer;
}

void free_attention(MultiHeadAttention *mha) {
  if (!mha)
    return;

  if (mha->Q_proj)
    NN_destroy(mha->Q_proj);
  if (mha->K_proj)
    NN_destroy(mha->K_proj);
  if (mha->V_proj)
    NN_destroy(mha->V_proj);
  if (mha->O_proj)
    NN_destroy(mha->O_proj);

  free(mha);
}

// Transformer destruction
void TRANSFORMER_destroy(Transformer_t *transformer) {
  if (!transformer)
    return;

  if (transformer->layers) {
    for (size_t i = 0; i < transformer->num_layers; i++) {
      if (transformer->layers[i]) {
        free_transformer_layer(transformer->layers[i]);
      }
    }
    free(transformer->layers);
  }
  free(transformer);
}

// Transformer forward pass for sequence
long double *TRANSFORMER_forward(Transformer_t *transformer,
                                 long double **input_sequence,
                                 size_t seq_length) {
  if (!transformer || !input_sequence || seq_length == 0)
    return NULL;

  size_t L = transformer->num_layers;
  size_t D = transformer->model_dim;

  // Start with the first input in the sequence
  long double *current = malloc(D * sizeof(long double));
  if (!current)
    return NULL;

  memcpy(current, input_sequence[0], D * sizeof(long double));

  // Process through all layers sequentially
  for (size_t i = 0; i < L; i++) {
    TransformerLayer *layer = transformer->layers[i];
    if (!layer) {
      free(current);
      return NULL;
    }

    // Set seq_length for this layer
    layer->seq_length = seq_length;

    // Forward pass through the layer (stores caches internally)
    long double *layer_output = transformer_forward(layer, current);
    if (!layer_output) {
      free(current);
      return NULL;
    }

    free(current);
    current = layer_output; // output becomes input for next layer
  }

  // Current now contains the model output
  return current;
}

// Transformer training
void TRANSFORMER_train(Transformer_t *transformer, long double **input_sequence,
                       size_t seq_length, long double *target) {
  if (!transformer || !input_sequence || !target)
    return;

  // Forward pass
  long double *output =
      TRANSFORMER_forward(transformer, input_sequence, seq_length);
  if (!output)
    return;

  // Calculate loss (simplified - just compute gradient)
  long double *grad_output =
      (long double *)malloc(transformer->model_dim * sizeof(long double));
  if (!grad_output) {
    free(output);
    return;
  }

  // Compute gradient (output - target)
  for (size_t i = 0; i < transformer->model_dim; i++) {
    grad_output[i] = output[i] - target[i];
  }

  // Backpropagate
  TRANSFORMER_backprop(transformer, input_sequence, seq_length, grad_output);

  free(grad_output);
  free(output);
}

// Transformer save
int TRANSFORMER_save(Transformer_t *transformer, FILE *file) {
  if (!transformer || !file)
    return 0;

  // Save transformer structure
  fwrite(&transformer->model_dim, sizeof(size_t), 1, file);
  fwrite(&transformer->num_heads, sizeof(size_t), 1, file);
  fwrite(&transformer->num_layers, sizeof(size_t), 1, file);

  // Save each layer (simplified - just save structure, not weights)
  for (size_t i = 0; i < transformer->num_layers; i++) {
    if (transformer->layers[i]) {
      fwrite(&transformer->layers[i]->model_dim, sizeof(size_t), 1, file);
      fwrite(&transformer->layers[i]->seq_length, sizeof(size_t), 1, file);
    }
  }

  return 1;
}

// Transformer load
Transformer_t *TRANSFORMER_load(FILE *file) {
  if (!file)
    return NULL;

  Transformer_t *transformer = (Transformer_t *)malloc(sizeof(Transformer_t));
  if (!transformer)
    return NULL;

  // Load transformer structure
  if (fread(&transformer->model_dim, sizeof(size_t), 1, file) != 1 ||
      fread(&transformer->num_heads, sizeof(size_t), 1, file) != 1 ||
      fread(&transformer->num_layers, sizeof(size_t), 1, file) != 1) {
    free(transformer);
    return NULL;
  }

  // Allocate layers
  transformer->layers = (TransformerLayer **)malloc(transformer->num_layers *
                                                    sizeof(TransformerLayer *));
  if (!transformer->layers) {
    free(transformer);
    return NULL;
  }

  // Load each layer (simplified - just recreate structure)
  size_t ff_dim = transformer->model_dim * 4;
  for (size_t i = 0; i < transformer->num_layers; i++) {
    size_t model_dim, seq_length;
    if (fread(&model_dim, sizeof(size_t), 1, file) != 1 ||
        fread(&seq_length, sizeof(size_t), 1, file) != 1) {
      // Cleanup on error
      for (size_t j = 0; j < i; j++) {
        free_transformer_layer(transformer->layers[j]);
      }
      free(transformer->layers);
      free(transformer);
      return NULL;
    }
    transformer->layers[i] =
        create_transformer_layer(model_dim, transformer->num_heads, ff_dim);
    if (!transformer->layers[i]) {
      // Cleanup on error
      for (size_t j = 0; j < i; j++) {
        free_transformer_layer(transformer->layers[j]);
      }
      free(transformer->layers);
      free(transformer);
      return NULL;
    }
    transformer->layers[i]->seq_length = seq_length;
  }

  return transformer;
}
