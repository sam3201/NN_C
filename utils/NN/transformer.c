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
long double **transformer_mha_forward(MultiHeadAttention *mha,
                                      long double **input_seq,
                                      size_t seq_length) {
  mha->seq_length = seq_length;
  size_t D = mha->head_dim;

  long double **Q = malloc(seq_length * sizeof(long double *));
  long double **K = malloc(seq_length * sizeof(long double *));
  long double **V = malloc(seq_length * sizeof(long double *));

  for (size_t t = 0; t < seq_length; t++) {
    Q[t] = NN_forward(mha->Q_proj, input_seq[t]);
    K[t] = NN_forward(mha->K_proj, input_seq[t]);
    V[t] = NN_forward(mha->V_proj, input_seq[t]);
  }

  mha->Q_cache = malloc(seq_length * D * sizeof(long double));
  mha->K_cache = malloc(seq_length * D * sizeof(long double));
  mha->V_cache = malloc(seq_length * D * sizeof(long double));

  for (size_t i = 0; i < seq_length; i++)
    for (size_t j = 0; j < D; j++) {
      mha->Q_cache[i * D + j] = Q[i][j];
      mha->K_cache[i * D + j] = K[i][j];
      mha->V_cache[i * D + j] = V[i][j];
    }

  long double *scores = calloc(seq_length * seq_length, sizeof(long double));

  for (size_t i = 0; i < seq_length; i++)
    for (size_t j = 0; j < seq_length; j++) {
      for (size_t k = 0; k < D; k++)
        scores[i * seq_length + j] +=
            mha->Q_cache[i * D + k] * mha->K_cache[j * D + k];
      scores[i * seq_length + j] /= sqrtl(D);
    }

  for (size_t i = 0; i < seq_length; i++) {
    long double max = scores[i * seq_length];
    for (size_t j = 1; j < seq_length; j++)
      if (scores[i * seq_length + j] > max)
        max = scores[i * seq_length + j];

    long double sum = 0;
    for (size_t j = 0; j < seq_length; j++) {
      scores[i * seq_length + j] = expl(scores[i * seq_length + j] - max);
      sum += scores[i * seq_length + j];
    }
    for (size_t j = 0; j < seq_length; j++)
      scores[i * seq_length + j] /= sum;
  }

  mha->scores_cache = scores;

  long double **att_out = malloc(seq_length * sizeof(long double *));

  for (size_t i = 0; i < seq_length; i++) {
    att_out[i] = calloc(D, sizeof(long double));
    for (size_t j = 0; j < D; j++)
      for (size_t k = 0; k < seq_length; k++)
        att_out[i][j] += scores[i * seq_length + k] * mha->V_cache[k * D + j];
  }

  long double **out = malloc(seq_length * sizeof(long double *));
  for (size_t t = 0; t < seq_length; t++)
    out[t] = NN_forward(mha->O_proj, att_out[t]);

  for (size_t t = 0; t < seq_length; t++) {
    free(att_out[t]);
    free(Q[t]);
    free(K[t]);
    free(V[t]);
  }
  free(att_out);
  free(Q);
  free(K);
  free(V);

  return out;
}

long double *transformer_norm_forward(LayerNorm *ln, long double *input) {
  size_t D = ln->dim;
  long double mean = 0, var = 0;

  for (size_t i = 0; i < D; i++)
    mean += input[i];
  mean /= D;

  for (size_t i = 0; i < D; i++) {
    long double d = input[i] - mean;
    var += d * d;
  }
  var /= D;

  long double *out = malloc(D * sizeof(long double));
  ln->input_cache = malloc(D * sizeof(long double));

  for (size_t i = 0; i < D; i++) {
    ln->input_cache[i] = input[i];
    out[i] = (input[i] - mean) / sqrtl(var + ln->epsilon);
  }

  return out;
}

void transformer_layernorm_backprop(LayerNorm *ln, long double *grad_output) {
  size_t D = ln->dim;

  long double mean = 0, var = 0;
  for (size_t i = 0; i < D; i++)
    mean += ln->input_cache[i];
  mean /= D;

  for (size_t i = 0; i < D; i++) {
    long double d = ln->input_cache[i] - mean;
    var += d * d;
  }
  var /= D;

  long double inv_std = 1.0L / sqrtl(var + ln->epsilon);

  for (size_t i = 0; i < D; i++) {
    grad_output[i] *= inv_std;
  }
}

void transformer_feedforward_backprop(FeedForward *ff,
                                      long double *grad_output) {
  NN_backprop_custom_delta(ff->network, grad_output);
}

void transformer_mha_backprop(MultiHeadAttention *mha,
                              long double *grad_output) {
  size_t T = mha->seq_length;
  size_t D = mha->head_dim;

  long double *dV = calloc(T * D, sizeof(long double));
  long double *dScores = calloc(T * T, sizeof(long double));

  for (size_t t = 0; t < T; t++)
    for (size_t k = 0; k < T; k++)
      for (size_t j = 0; j < D; j++)
        dScores[t * T + k] += grad_output[t * D + j] * mha->V_cache[k * D + j];

  for (size_t t = 0; t < T; t++) {
    for (size_t i = 0; i < T; i++) {
      for (size_t j = 0; j < T; j++) {
        long double s_i = mha->scores_cache[t * T + i];
        long double s_j = mha->scores_cache[t * T + j];
        long double delta = (i == j);
        dScores[t * T + i] += dScores[t * T + j] * s_i * (delta - s_j);
      }
    }
  }
  long double *dQ = calloc(T * D, sizeof(long double));
  long double *dK = calloc(T * D, sizeof(long double));

  for (size_t t = 0; t < T; t++)
    for (size_t k = 0; k < T; k++)
      for (size_t j = 0; j < D; j++) {
        long double g = dScores[t * T + k] / sqrtl(D);
        dQ[t * D + j] += g * mha->K_cache[k * D + j];
        dK[k * D + j] += g * mha->Q_cache[t * D + j];
      }

  for (size_t t = 0; t < T; t++) {
    NN_backprop(mha->Q_proj, &dQ[t * D]);
    NN_backprop(mha->K_proj, &dK[t * D]);
    NN_backprop(mha->V_proj, &dV[t * D]);
  }

  free(dV);
  free(dScores);
  free(dQ);
  free(dK);
}

// ----------------------
// Transformer forward and backprop
// ----------------------
long double **transformer_layer_forward(TransformerLayer *layer,
                                        long double **input,
                                        size_t seq_length) {
  size_t D = layer->model_dim;

  // Cache input for backprop
  layer->attention_input = malloc(seq_length * D * sizeof(long double));
  for (size_t t = 0; t < seq_length; t++)
    memcpy(&layer->attention_input[t * D], input[t], D * sizeof(long double));

  // ---- Multi-head attention ----
  long double **att =
      transformer_mha_forward(layer->attention, input, seq_length);

  // ---- Residual + norm1 ----
  long double **norm1_out = malloc(seq_length * sizeof(long double *));
  for (size_t t = 0; t < seq_length; t++) {
    for (size_t i = 0; i < D; i++)
      att[t][i] += input[t][i];
    norm1_out[t] = transformer_norm_forward(layer->norm1, att[t]);
    free(att[t]);
  }
  free(att);

  // ---- Feed-forward ----
  long double **ff_out = malloc(seq_length * sizeof(long double *));
  for (size_t t = 0; t < seq_length; t++) {
    layer->feed_forward->input_cache = norm1_out[t];
    ff_out[t] = NN_forward(layer->feed_forward->network, norm1_out[t]);
  }

  // ---- Residual + norm2 ----
  long double **out = malloc(seq_length * sizeof(long double *));
  for (size_t t = 0; t < seq_length; t++) {
    for (size_t i = 0; i < D; i++)
      ff_out[t][i] += norm1_out[t][i];
    out[t] = transformer_norm_forward(layer->norm2, ff_out[t]);
    free(ff_out[t]);
    free(norm1_out[t]);
  }

  free(ff_out);
  free(norm1_out);
  return out;
}

long double **TRANSFORMER_forward(Transformer_t *transformer,
                                  long double **input_sequence,
                                  size_t seq_length) {
  long double **x = input_sequence;

  for (size_t l = 0; l < transformer->num_layers; l++) {
    x = transformer_layer_forward(transformer->layers[l], x, seq_length);
  }

  return x;
}

// ----------------------
// Backprop
// ----------------------
long double *transformer_layer_backprop(TransformerLayer *layer,
                                        long double *grad_output) {
  size_t D = layer->model_dim;

  // ---- Norm2 ----
  transformer_layernorm_backprop(layer->norm2, grad_output);

  // ---- Residual add ----
  long double *grad_ff = malloc(D * sizeof(long double));
  memcpy(grad_ff, grad_output, D * sizeof(long double));

  // ---- Feed-forward ----
  transformer_feedforward_backprop(layer->feed_forward, grad_ff);

  // ---- Norm1 ----
  transformer_layernorm_backprop(layer->norm1, grad_ff);

  // ---- Attention ----
  transformer_mha_backprop(layer->attention, grad_ff);

  return grad_ff; // gradient to previous layer
}
long double **TRANSFORMER_backprop(Transformer_t *transformer,
                                   long double **grad_output,
                                   size_t seq_length) {
  long double **grad = grad_output;

  for (ssize_t l = transformer->num_layers - 1; l >= 0; l--) {
    // TODO: real layer backprop
    // For now, pass through
    grad = grad;
  }

  return grad;
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
