#include "TRANSFORMER.h"
#include <math.h>
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

  mha->Q_cache = NULL;
  mha->K_cache = NULL;
  mha->V_cache = NULL;
  mha->scores_cache = NULL;

  if (!mha->Q_proj || !mha->K_proj || !mha->V_proj || !mha->O_proj) {
    free_attention(mha);
    return NULL;
  }

  return mha;
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

// ----------------------
// Free memory
// ----------------------
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

  // Flatten caches for compatibility
  mha->Q_cache = Q[0];
  mha->K_cache = K[0];
  mha->V_cache = V[0];

  long double *scores = malloc(seq_length * seq_length * sizeof(long double));
  if (!scores)
    return NULL;

  for (size_t i = 0; i < seq_length; i++) {
    for (size_t j = 0; j < seq_length; j++) {
      scores[i * seq_length + j] = 0;
      for (size_t k = 0; k < mha->head_dim; k++) {
        scores[i * seq_length + j] += mha->Q_cache[i * mha->head_dim + k] *
                                      mha->K_cache[j * mha->head_dim + k];
      }
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

  long double **final_out = NN_forward(mha->O_proj, att_out);
  free(att_out);
  return final_out[0];
}

// LayerNorm forward
long double *transformer_norm_forward(LayerNorm *ln, long double *input) {
  if (!ln || !input)
    return NULL;

  ln->input_cache = malloc(ln->dim * sizeof(long double));
  memcpy(ln->input_cache, input, ln->dim * sizeof(long double));

  long double mean = 0;
  for (size_t i = 0; i < ln->dim; i++)
    mean += input[i];
  mean /= ln->dim;

  long double var = 0;
  for (size_t i = 0; i < ln->dim; i++)
    var += (input[i] - mean) * (input[i] - mean);
  var /= ln->dim;

  long double *normalized = malloc(ln->dim * sizeof(long double));
  for (size_t i = 0; i < ln->dim; i++)
    normalized[i] = (input[i] - mean) / sqrtl(var + ln->epsilon);

  long double **output = NN_forward(ln->norm_network, normalized);
  free(normalized);
  return output[0];
}

// ----------------------
// Backprop & Training
// ----------------------
void TRANSFORMER_backprop(Transformer_t *transformer,
                          long double **input_sequence, size_t seq_length,
                          long double *grad_loss) {
  if (!transformer || !input_sequence || !grad_loss)
    return;

  long double *grad = grad_loss;
  for (ssize_t l = transformer->num_layers - 1; l >= 0; l--) {
    transformer_layer_backprop(transformer->layers[l], grad);
  }
}

void TRANSFORMER_train(Transformer_t *transformer, long double **input_sequence,
                       size_t seq_length, long double *target) {
  if (!transformer || !input_sequence || !target)
    return;

  long double *output =
      TRANSFORMER_forward(transformer, input_sequence, seq_length);
  if (!output)
    return;

  long double *grad_loss = malloc(transformer->model_dim * sizeof(long double));
  for (size_t i = 0; i < transformer->model_dim; i++)
    grad_loss[i] = output[i] - target[i];

  TRANSFORMER_backprop(transformer, input_sequence, seq_length, grad_loss);

  free(output);
  free(grad_loss);
}

// ----------------------
// Serialization
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

    NN_save(layer->attention->Q_proj, file);
    NN_save(layer->attention->K_proj, file);
    NN_save(layer->attention->V_proj, file);
    NN_save(layer->attention->O_proj, file);

    NN_save(layer->feed_forward->network, file);
    NN_save(layer->norm1->norm_network, file);
    NN_save(layer->norm2->norm_network, file);
  }
  return 1;
}

Transformer_t *TRANSFORMER_load(FILE *file) {
  if (!file)
    return NULL;

  Transformer_t *transformer = malloc(sizeof(Transformer_t));
  if (!transformer)
    return NULL;

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

    NN_load(layer->attention->Q_proj, file);
    NN_load(layer->attention->K_proj, file);
    NN_load(layer->attention->V_proj, file);
    NN_load(layer->attention->O_proj, file);

    NN_load(layer->feed_forward->network, file);
    NN_load(layer->norm1->norm_network, file);
    NN_load(layer->norm2->norm_network, file);

    transformer->layers[i] = layer;
  }

  return transformer;
}
