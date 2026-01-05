#include "TRANSFORMER.h"
#include "NN.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// ----------------------
// Creation functions
// ----------------------
MultiHeadAttention *create_attention(size_t model_dim, size_t num_heads) {
  MultiHeadAttention *mha = malloc(sizeof(MultiHeadAttention));
  if (!mha)
    return NULL;

  mha->model_dim = model_dim;
  mha->num_heads = num_heads;
  mha->head_dim = model_dim / num_heads;
  mha->seq_length = 0;

  size_t layers[] = {model_dim, mha->head_dim, 0};
  ActivationFunctionType acts[] = {RELU, RELU};
  ActivationDerivativeType ders[] = {RELU_DERIVATIVE, RELU_DERIVATIVE};

  mha->Q_proj =
      NN_init(layers, acts, ders, MSE, MSE_DERIVATIVE, L1, SGD, 0.01L);
  mha->K_proj =
      NN_init(layers, acts, ders, MSE, MSE_DERIVATIVE, L1, SGD, 0.01L);
  mha->V_proj =
      NN_init(layers, acts, ders, MSE, MSE_DERIVATIVE, L1, SGD, 0.01L);
  mha->O_proj =
      NN_init(layers, acts, ders, MSE, MSE_DERIVATIVE, L1, SGD, 0.01L);

  mha->Q_cache = mha->K_cache = mha->V_cache = mha->scores_cache = NULL;

  if (!mha->Q_proj || !mha->K_proj || !mha->V_proj || !mha->O_proj) {
    free_attention(mha);
    return NULL;
  }
  return mha;
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
  if (mha->Q_cache)
    free(mha->Q_cache);
  if (mha->K_cache)
    free(mha->K_cache);
  if (mha->V_cache)
    free(mha->V_cache);
  if (mha->scores_cache)
    free(mha->scores_cache);
  free(mha);
}

FeedForward *create_feed_forward(size_t input_dim, size_t hidden_dim) {
  FeedForward *ff = malloc(sizeof(FeedForward));
  if (!ff)
    return NULL;

  ff->input_dim = input_dim;
  ff->hidden_dim = hidden_dim;

  size_t layers[] = {input_dim, hidden_dim, input_dim, 0};
  ActivationFunctionType acts[] = {RELU, RELU, RELU};
  ActivationDerivativeType ders[] = {RELU_DERIVATIVE, RELU_DERIVATIVE,
                                     RELU_DERIVATIVE};

  ff->network =
      NN_init(layers, acts, ders, MSE, MSE_DERIVATIVE, L1, SGD, 0.01L);
  ff->input_cache = NULL;

  if (!ff->network) {
    free(ff);
    return NULL;
  }
  return ff;
}

LayerNorm *create_layer_norm(size_t dim, long double epsilon) {
  LayerNorm *ln = malloc(sizeof(LayerNorm));
  if (!ln)
    return NULL;

  ln->dim = dim;
  ln->epsilon = epsilon;

  size_t layers[] = {dim, dim, 0};
  ActivationFunctionType acts[] = {RELU, RELU};
  ActivationDerivativeType ders[] = {RELU_DERIVATIVE, RELU_DERIVATIVE};

  ln->norm_network =
      NN_init(layers, acts, ders, MSE, MSE_DERIVATIVE, L1, SGD, 0.01L);
  ln->input_cache = NULL;

  if (!ln->norm_network) {
    free(ln);
    return NULL;
  }
  return ln;
}

TransformerLayer *create_transformer_layer(size_t model_dim, size_t num_heads,
                                           size_t ff_dim) {
  TransformerLayer *layer = malloc(sizeof(TransformerLayer));
  if (!layer)
    return NULL;

  layer->model_dim = model_dim;
  layer->seq_length = 1;

  layer->attention = create_attention(model_dim, num_heads);
  layer->norm1 = create_layer_norm(model_dim, 1e-6L);
  layer->feed_forward = create_feed_forward(model_dim, ff_dim);
  layer->norm2 = create_layer_norm(model_dim, 1e-6L);

  if (!layer->attention || !layer->norm1 || !layer->feed_forward ||
      !layer->norm2) {
    free_transformer_layer(layer);
    return NULL;
  }

  layer->attention_input = NULL;
  layer->norm1_input = NULL;
  layer->ff_input = NULL;
  layer->norm2_input = NULL;

  return layer;
}

void free_feed_forward(FeedForward *ff) {
  if (!ff)
    return;
  if (ff->network)
    NN_destroy(ff->network);
  if (ff->input_cache)
    free(ff->input_cache);
  free(ff);
}
void free_layer_norm(LayerNorm *ln) {
  if (!ln)
    return;
  if (ln->norm_network)
    NN_destroy(ln->norm_network);
  if (ln->input_cache)
    free(ln->input_cache);
  free(ln);
}
void free_transformer_layer(TransformerLayer *layer) {
  if (!layer)
    return;
  free_attention(layer->attention);
  free_layer_norm(layer->norm1);
  free_feed_forward(layer->feed_forward);
  free_layer_norm(layer->norm2);
  if (layer->attention_input)
    free(layer->attention_input);
  if (layer->norm1_input)
    free(layer->norm1_input);
  if (layer->ff_input)
    free(layer->ff_input);
  if (layer->norm2_input)
    free(layer->norm2_input);
  free(layer);
}
void TRANSFORMER_destroy(Transformer_t *transformer) {
  if (!transformer)
    return;
  for (size_t i = 0; i < transformer->num_layers; i++)
    free_transformer_layer(transformer->layers[i]);
  free(transformer->layers);
  free(transformer);
}

// ----------------------
// Forward pass
// ----------------------
long double *transformer_mha_forward(MultiHeadAttention *mha,
                                     long double *input, size_t seq_length) {
  if (!mha || !input)
    return NULL;
  mha->seq_length = seq_length;

  long double **Q = NN_forward(mha->Q_proj, input);
  long double **K = NN_forward(mha->K_proj, input);
  long double **V = NN_forward(mha->V_proj, input);

  // Flatten outputs for caching
  mha->Q_cache = malloc(seq_length * mha->head_dim * sizeof(long double));
  mha->K_cache = malloc(seq_length * mha->head_dim * sizeof(long double));
  mha->V_cache = malloc(seq_length * mha->head_dim * sizeof(long double));
  for (size_t i = 0; i < seq_length; i++)
    for (size_t j = 0; j < mha->head_dim; j++) {
      mha->Q_cache[i * mha->head_dim + j] = Q[i][j];
      mha->K_cache[i * mha->head_dim + j] = K[i][j];
      mha->V_cache[i * mha->head_dim + j] = V[i][j];
    }

  // Compute attention scores
  long double *scores = malloc(seq_length * seq_length * sizeof(long double));
  for (size_t i = 0; i < seq_length; i++) {
    for (size_t j = 0; j < seq_length; j++) {
      scores[i * seq_length + j] = 0;
      for (size_t k = 0; k < mha->head_dim; k++)
        scores[i * seq_length + j] += mha->Q_cache[i * mha->head_dim + k] *
                                      mha->K_cache[j * mha->head_dim + k];
      scores[i * seq_length + j] /= sqrt(mha->head_dim);
    }
  }

  // Softmax
  for (size_t i = 0; i < seq_length; i++) {
    long double max_val = scores[i * seq_length];
    for (size_t j = 1; j < seq_length; j++)
      if (scores[i * seq_length + j] > max_val)
        max_val = scores[i * seq_length + j];
    long double sum = 0;
    for (size_t j = 0; j < seq_length; j++) {
      scores[i * seq_length + j] = expl(scores[i * seq_length + j] - max_val);
      sum += scores[i * seq_length + j];
    }
    for (size_t j = 0; j < seq_length; j++)
      scores[i * seq_length + j] /= sum;
  }
  mha->scores_cache = scores;

  long double *att_out =
      malloc(seq_length * mha->head_dim * sizeof(long double));
  for (size_t i = 0; i < seq_length; i++)
    for (size_t j = 0; j < mha->head_dim; j++) {
      att_out[i * mha->head_dim + j] = 0;
      for (size_t k = 0; k < seq_length; k++)
        att_out[i * mha->head_dim + j] +=
            scores[i * seq_length + k] * mha->V_cache[k * mha->head_dim + j];
    }

  long double *final_out =
      malloc(seq_length * mha->head_dim * sizeof(long double));
  long double **O_out = NN_forward(mha->O_proj, att_out);
  for (size_t i = 0; i < seq_length; i++)
    for (size_t j = 0; j < mha->head_dim; j++)
      final_out[i * mha->head_dim + j] = O_out[i][j];
  free(att_out);

  return final_out;
}

// ----------------------
// Transformer forward and backprop
// ----------------------
long double *TRANSFORMER_forward(Transformer_t *transformer,
                                 long double **input_sequence,
                                 size_t seq_length) {
  long double *x = input_sequence[0];
  for (size_t l = 0; l < transformer->num_layers; l++)
    x = transformer_layer_forward(transformer->layers[l], x);
  return x;
}

// ----------------------
// MultiHeadAttention backprop
// ----------------------
long double *transformer_mha_backprop(MultiHeadAttention *mha,
                                      long double *grad_output,
                                      size_t seq_length) {
  if (!mha || !grad_output)
    return NULL;

  size_t D = mha->head_dim;
  long double *grad_att_out = malloc(seq_length * D * sizeof(long double));
  memset(grad_att_out, 0, seq_length * D * sizeof(long double));

  // Backprop through O_proj
  long double **grad_O =
      NN_backprop(mha->O_proj, grad_output); // returns long double **
  for (size_t i = 0; i < seq_length; i++)
    for (size_t j = 0; j < D; j++)
      grad_att_out[i * D + j] = grad_O[i][j];

  // Backprop through attention scores: grad_att_out = dLoss/d(att_out)
  long double *grad_scores =
      malloc(seq_length * seq_length * sizeof(long double));
  memset(grad_scores, 0, seq_length * seq_length * sizeof(long double));

  for (size_t i = 0; i < seq_length; i++)
    for (size_t j = 0; j < seq_length; j++)
      for (size_t k = 0; k < D; k++)
        grad_scores[i * seq_length + j] +=
            grad_att_out[i * D + k] * mha->V_cache[j * D + k];

  // Backprop through softmax
  for (size_t i = 0; i < seq_length; i++) {
    long double *row = &mha->scores_cache[i * seq_length];
    long double *grad_row = &grad_scores[i * seq_length];
    long double sum = 0;
    for (size_t j = 0; j < seq_length; j++)
      sum += row[j] * grad_row[j];
    for (size_t j = 0; j < seq_length; j++)
      grad_row[j] = row[j] * (grad_row[j] - sum);
  }

  // Backprop through V_proj
  long double *grad_V = malloc(seq_length * D * sizeof(long double));
  memset(grad_V, 0, seq_length * D * sizeof(long double));
  for (size_t i = 0; i < seq_length; i++)
    for (size_t j = 0; j < D; j++)
      for (size_t k = 0; k < seq_length; k++)
        grad_V[i * D + j] +=
            mha->scores_cache[k * seq_length + i] * grad_att_out[k * D + j];

  long double **grad_V_out = NN_backprop(mha->V_proj, grad_V);

  // Backprop through Q_proj
  long double *grad_Q = malloc(seq_length * D * sizeof(long double));
  memset(grad_Q, 0, seq_length * D * sizeof(long double));
  for (size_t i = 0; i < seq_length; i++)
    for (size_t j = 0; j < D; j++)
      for (size_t k = 0; k < seq_length; k++)
        grad_Q[i * D + j] +=
            grad_scores[i * seq_length + k] * mha->K_cache[k * D + j];

  long double **grad_Q_out = NN_backprop(mha->Q_proj, grad_Q);

  // Backprop through K_proj
  long double *grad_K = malloc(seq_length * D * sizeof(long double));
  memset(grad_K, 0, seq_length * D * sizeof(long double));
  for (size_t i = 0; i < seq_length; i++)
    for (size_t j = 0; j < D; j++)
      for (size_t k = 0; k < seq_length; k++)
        grad_K[i * D + j] +=
            grad_scores[k * seq_length + i] * mha->Q_cache[k * D + j];

  long double **grad_K_out = NN_backprop(mha->K_proj, grad_K);

  // Sum gradients from Q, K, V paths
  long double *grad_input = malloc(seq_length * D * sizeof(long double));
  memset(grad_input, 0, seq_length * D * sizeof(long double));
  for (size_t i = 0; i < seq_length; i++)
    for (size_t j = 0; j < D; j++)
      grad_input[i * D + j] =
          grad_Q_out[i][j] + grad_K_out[i][j] + grad_V_out[i][j];

  // Free temporary arrays
  free(grad_att_out);
  free(grad_scores);
  free(grad_V);
  free(grad_Q);
  free(grad_K);

  return grad_input;
}

// ----------------------
// Transformer Layer backprop
// ----------------------
void transformer_layer_backprop(TransformerLayer *layer,
                                long double *grad_output) {
  if (!layer || !grad_output)
    return;

  // Backprop through second LayerNorm (Add & Norm2)
  long double *grad_ff_out = layer_norm_backprop(layer->norm2, grad_output);

  // Backprop through FeedForward
  long double *grad_ff_input =
      NN_backprop(layer->feed_forward->network, grad_ff_out);

  // Backprop through first LayerNorm (Add & Norm1)
  long double *grad_att_out = layer_norm_backprop(layer->norm1, grad_ff_input);

  // Backprop through MHA
  long double *grad_input = transformer_mha_backprop(
      layer->attention, grad_att_out, layer->seq_length);

  // Free intermediate grads
  free(grad_ff_out);
  free(grad_ff_input);
  free(grad_att_out);

  // grad_input is the final gradient w.r.t. the layer input
  // Could be returned if stacking multiple layers
}

// ----------------------
// Simple LayerNorm backprop helper
// ----------------------
long double *layer_norm_backprop(LayerNorm *ln, long double *grad_output) {
  if (!ln || !grad_output)
    return NULL;
  return NN_backprop(ln->norm_network, grad_output);
}

// Backprop
// ----------------------
// Full Transformer backprop over a sequence
// ----------------------
// ----------------------
// Full Transformer backprop over a sequence
// ----------------------
void TRANSFORMER_backprop(Transformer_t *transformer,
                          long double **input_sequence, size_t seq_length,
                          long double *grad_loss) {
  if (!transformer || !input_sequence || !grad_loss)
    return;

  size_t model_dim = transformer->model_dim;

  // Allocate gradient buffer for each time step
  long double **grad_step = malloc(seq_length * sizeof(long double *));
  for (size_t t = 0; t < seq_length; t++) {
    grad_step[t] = malloc(model_dim * sizeof(long double));
    // For the last time step, initialize from grad_loss
    if (t == seq_length - 1) {
      memcpy(grad_step[t], grad_loss, model_dim * sizeof(long double));
    } else {
      memset(grad_step[t], 0, model_dim * sizeof(long double));
    }
  }

  // Backprop through layers in reverse order
  for (ssize_t l = transformer->num_layers - 1; l >= 0; l--) {
    TransformerLayer *layer = transformer->layers[l];

    // Iterate backward through the sequence
    for (ssize_t t = seq_length - 1; t >= 0; t--) {
      long double *grad_in = transformer_layer_backprop(layer, grad_step[t]);

      // Accumulate gradient into previous time step if not first
      if (t > 0) {
        for (size_t i = 0; i < model_dim; i++)
          grad_step[t - 1][i] += grad_in[i];
      }

      free(grad_in);
    }
  }

  // Free sequence gradient buffers
  for (size_t t = 0; t < seq_length; t++)
    free(grad_step[t]);
  free(grad_step);
}

// ----------------------
// Save/load transformer
// ----------------------
int TRANSFORMER_save(Transformer_t *transformer, FILE *file) {
  if (!transformer || !file)
    return 0;
  fwrite(&transformer->model_dim, sizeof(size_t), 1, file);
  fwrite(&transformer->num_heads, sizeof(size_t), 1, file);
  fwrite(&transformer->num_layers, sizeof(size_t), 1, file);
  for (size_t i = 0; i < transformer->num_layers; i++) {
    TransformerLayer *layer = transformer->layers[i];
    fwrite(&layer->model_dim, sizeof(size_t), 1, file);
    fwrite(&layer->seq_length, sizeof(size_t), 1, file);
    NN_save(layer->attention->Q_proj, "Q_proj.nn");
    NN_save(layer->attention->K_proj, "K_proj.nn");
    NN_save(layer->attention->V_proj, "V_proj.nn");
    NN_save(layer->attention->O_proj, "O_proj.nn");
    NN_save(layer->feed_forward->network, "ff.nn");
    NN_save(layer->norm1->norm_network, "norm1.nn");
    NN_save(layer->norm2->norm_network, "norm2.nn");
  }
  return 1;
}

Transformer_t *TRANSFORMER_load(FILE *file) {
  if (!file)
    return NULL;
  Transformer_t *transformer = malloc(sizeof(Transformer_t));
  fread(&transformer->model_dim, sizeof(size_t), 1, file);
  fread(&transformer->num_heads, sizeof(size_t), 1, file);
  fread(&transformer->num_layers, sizeof(size_t), 1, file);
  transformer->layers =
      malloc(transformer->num_layers * sizeof(TransformerLayer *));
  size_t ff_dim = transformer->model_dim * 4;
  for (size_t i = 0; i < transformer->num_layers; i++) {
    TransformerLayer *layer = create_transformer_layer(
        transformer->model_dim, transformer->num_heads, ff_dim);
    fread(&layer->model_dim, sizeof(size_t), 1, file);
    fread(&layer->seq_length, sizeof(size_t), 1, file);
    // load NNs from file names
    layer->attention->Q_proj = NN_load("Q_proj.nn");
    layer->attention->K_proj = NN_load("K_proj.nn");
    layer->attention->V_proj = NN_load("V_proj.nn");
    layer->attention->O_proj = NN_load("O_proj.nn");
    layer->feed_forward->network = NN_load("ff.nn");
    layer->norm1->norm_network = NN_load("norm1.nn");
    layer->norm2->norm_network = NN_load("norm2.nn");
    transformer->layers[i] = layer;
  }
  return transformer;
}
