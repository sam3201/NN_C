#include "TRANSFORMER.h"
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

// Free memory
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

// Training
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
// Forward pass
// ----------------------
long double *transformer_mha_forward(MultiHeadAttention *mha,
                                     long double *input, size_t seq_length) {
  if (!mha || !input)
    return NULL;

  mha->seq_length = seq_length;

  // Project input
  long double *Q = NN_forward(mha->Q_proj, input);
  long double *K = NN_forward(mha->K_proj, input);
  long double *V = NN_forward(mha->V_proj, input);

  // Cache
  mha->Q_cache = Q;
  mha->K_cache = K;
  mha->V_cache = V;

  long double *scores = malloc(seq_length * seq_length * sizeof(long double));
  if (!scores)
    return NULL;

  // Q*K^T / sqrt(dk)
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

  // Softmax
  for (size_t i = 0; i < seq_length; i++) {
    long double max_val = scores[i * seq_length];
    for (size_t j = 1; j < seq_length; j++)
      if (scores[i * seq_length + j] > max_val)
        max_val = scores[i * seq_length + j];

    long double sum = 0;
    for (size_t j = 0; j < seq_length; j++) {
      scores[i * seq_length + j] = exp(scores[i * seq_length + j] - max_val);
      sum += scores[i * seq_length + j];
    }
    for (size_t j = 0; j < seq_length; j++)
      scores[i * seq_length + j] /= sum;
  }

  mha->scores_cache = scores;

  // Attention output
  long double *att_out =
      malloc(seq_length * mha->head_dim * sizeof(long double));
  for (size_t i = 0; i < seq_length; i++)
    for (size_t j = 0; j < mha->head_dim; j++) {
      att_out[i * mha->head_dim + j] = 0;
      for (size_t k = 0; k < seq_length; k++)
        att_out[i * mha->head_dim + j] +=
            scores[i * seq_length + k] * V[k * mha->head_dim + j];
    }

  long double *final_out = NN_forward(mha->O_proj, att_out);
  free(att_out);

  return final_out;
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
    normalized[i] = (input[i] - mean) / sqrt(var + ln->epsilon);

  long double *output = NN_forward(ln->norm_network, normalized);
  free(normalized);

  return output;
}

// Transformer layer forward
long double *transformer_layer_forward(TransformerLayer *layer,
                                       long double *input) {
  if (!layer || !input)
    return NULL;

  layer->attention_input = input;

  // MHA + Add & Norm
  long double *att_out =
      transformer_mha_forward(layer->attention, input, layer->seq_length);
  long double *res1 = malloc(layer->model_dim * sizeof(long double));
  for (size_t i = 0; i < layer->model_dim; i++)
    res1[i] = input[i] + att_out[i];
  free(att_out);

  layer->norm1_input = res1;
  long double *norm1_out = transformer_norm_forward(layer->norm1, res1);

  // Feed-forward + Add & Norm
  layer->ff_input = norm1_out;
  long double *ff_out = NN_forward(layer->feed_forward->network, norm1_out);
  free(norm1_out);

  long double *res2 = malloc(layer->model_dim * sizeof(long double));
  for (size_t i = 0; i < layer->model_dim; i++)
    res2[i] = layer->ff_input[i] + ff_out[i];
  free(ff_out);

  layer->norm2_input = res2;
  long double *output = transformer_norm_forward(layer->norm2, res2);
  free(res2);

  return output;
}

long double *TRANSFORMER_forward(Transformer_t *transformer,
                                 long double **input_sequence,
                                 size_t seq_length) {
  if (!transformer || !input_sequence)
    return NULL;

  long double *x = input_sequence[0]; // Assuming seq_length vectors
  for (size_t l = 0; l < transformer->num_layers; l++) {
    x = transformer_layer_forward(transformer->layers[l], x);
  }
  return x; // caller must free
}

// ----------------------
// Backpropagation
// ----------------------

// Multi-head attention backprop
void transformer_mha_backprop(MultiHeadAttention *mha,
                              long double *grad_output) {
  if (!mha || !grad_output)
    return;

  size_t seq = mha->seq_length;
  size_t d = mha->head_dim;

  // Step 1: Backprop through output projection
  NN_backprop(mha->O_proj, mha->V_cache, grad_output, 0.01L);

  // Step 2: Grad w.r.t attention scores (simplified for single head)
  long double *grad_scores = malloc(seq * seq * sizeof(long double));
  memset(grad_scores, 0, seq * seq * sizeof(long double));

  // dL/dscores = grad_output * V^T
  for (size_t i = 0; i < seq; i++) {
    for (size_t j = 0; j < seq; j++) {
      for (size_t k = 0; k < d; k++) {
        grad_scores[i * seq + j] +=
            grad_output[i * d + k] * mha->V_cache[j * d + k];
      }
    }
  }

  // Step 3: Softmax gradient
  for (size_t i = 0; i < seq; i++) {
    long double sum = 0;
    for (size_t j = 0; j < seq; j++)
      sum += grad_scores[i * seq + j] * mha->scores_cache[i * seq + j];

    for (size_t j = 0; j < seq; j++)
      grad_scores[i * seq + j] =
          mha->scores_cache[i * seq + j] * (grad_scores[i * seq + j] - sum);
  }

  // Step 4: Grad w.r.t Q, K, V projections
  NN_backprop(mha->Q_proj, mha->Q_cache, grad_scores, 0.01L);
  NN_backprop(mha->K_proj, mha->K_cache, grad_scores, 0.01L);
  NN_backprop(mha->V_proj, mha->V_cache, grad_output, 0.01L);

  free(grad_scores);

  // Clear caches
  free(mha->Q_cache);
  free(mha->K_cache);
  free(mha->V_cache);
  free(mha->scores_cache);
  mha->Q_cache = mha->K_cache = mha->V_cache = mha->scores_cache = NULL;
}

// LayerNorm backprop
void transformer_layer_backprop(TransformerLayer *layer,
                                long double *grad_output) {
  if (!layer || !grad_output)
    return;

  // Step 1: Backprop through second layer norm
  transformer_layernorm_backprop(layer->norm2, grad_output);

  // Step 2: Backprop through feed-forward
  transformer_feedforward_backprop(layer->feed_forward, layer->norm2_input);

  // Step 3: Add residual gradient from feed-forward input
  for (size_t i = 0; i < layer->model_dim; i++)
    layer->norm1_input[i] += layer->ff_input[i];

  // Step 4: Backprop through first layer norm
  transformer_layernorm_backprop(layer->norm1, layer->norm1_input);

  // Step 5: Backprop through multi-head attention
  transformer_mha_backprop(layer->attention, layer->attention_input);
}

// Backprop through entire transformer
void TRANSFORMER_backprop(Transformer_t *transformer,
                          long double **input_sequence, size_t seq_length,
                          long double *grad_loss) {
  if (!transformer || !input_sequence || !grad_loss)
    return;

  long double *grad = grad_loss;
  // Iterate layers backwards
  for (ssize_t l = transformer->num_layers - 1; l >= 0; l--) {
    transformer_layer_backprop(transformer->layers[l], grad);
  }
}

// Training
void TRANSFORMER_train(Transformer_t *transformer, long double **input_sequence,
                       size_t seq_length, long double *target) {
  if (!transformer || !input_sequence || !target)
    return;

  // Forward pass
  long double *output =
      TRANSFORMER_forward(transformer, input_sequence, seq_length);
  if (!output)
    return;

  // Compute gradient w.r.t MSE
  long double *grad_loss = malloc(transformer->model_dim * sizeof(long double));
  for (size_t i = 0; i < transformer->model_dim; i++)
    grad_loss[i] = output[i] - target[i];

  // Backprop
  TRANSFORMER_backprop(transformer, input_sequence, seq_length, grad_loss);

  free(output);
  free(grad_loss);
}

// Save transformer to file
int TRANSFORMER_save(Transformer_t *transformer, const char *filename) {
  if (!transformer || !filename)
    return 0;

  FILE *file = fopen(filename, "wb");
  if (!file)
    return 0;

  // Save global info
  fwrite(&transformer->model_dim, sizeof(size_t), 1, file);
  fwrite(&transformer->num_heads, sizeof(size_t), 1, file);
  fwrite(&transformer->num_layers, sizeof(size_t), 1, file);

  // Save each layer
  for (size_t i = 0; i < transformer->num_layers; i++) {
    TransformerLayer *layer = transformer->layers[i];
    if (!layer) {
      fclose(file);
      return 0;
    }

    fwrite(&layer->model_dim, sizeof(size_t), 1, file);
    fwrite(&layer->seq_length, sizeof(size_t), 1, file);

    // Save NNs with NN_save (uses filename internally)
    NN_save(layer->attention->Q_proj, filename);
    NN_save(layer->attention->K_proj, filename);
    NN_save(layer->attention->V_proj, filename);
    NN_save(layer->attention->O_proj, filename);

    NN_save(layer->feed_forward->network, filename);

    NN_save(layer->norm1->norm_network, filename);
    NN_save(layer->norm2->norm_network, filename);
  }

  fclose(file);
  return 1;
}

// Load transformer from file
Transformer_t *TRANSFORMER_load(const char *filename) {
  if (!filename)
    return NULL;

  FILE *file = fopen(filename, "rb");
  if (!file)
    return NULL;

  Transformer_t *transformer = malloc(sizeof(Transformer_t));
  if (!transformer) {
    fclose(file);
    return NULL;
  }

  // Load global info
  if (fread(&transformer->model_dim, sizeof(size_t), 1, file) != 1 ||
      fread(&transformer->num_heads, sizeof(size_t), 1, file) != 1 ||
      fread(&transformer->num_layers, sizeof(size_t), 1, file) != 1) {
    free(transformer);
    fclose(file);
    return NULL;
  }

  transformer->layers =
      malloc(transformer->num_layers * sizeof(TransformerLayer *));
  if (!transformer->layers) {
    free(transformer);
    fclose(file);
    return NULL;
  }

  size_t ff_dim = transformer->model_dim * 4;

  for (size_t i = 0; i < transformer->num_layers; i++) {
    TransformerLayer *layer = create_transformer_layer(
        transformer->model_dim, transformer->num_heads, ff_dim);
    if (!layer) {
      // Cleanup previous layers
      for (size_t j = 0; j < i; j++)
        free_transformer_layer(transformer->layers[j]);
      free(transformer->layers);
      free(transformer);
      fclose(file);
      return NULL;
    }

    // Load layer metadata
    if (fread(&layer->model_dim, sizeof(size_t), 1, file) != 1 ||
        fread(&layer->seq_length, sizeof(size_t), 1, file) != 1) {
      free_transformer_layer(layer);
      for (size_t j = 0; j < i; j++)
        free_transformer_layer(transformer->layers[j]);
      free(transformer->layers);
      free(transformer);
      fclose(file);
      return NULL;
    }

    // Load NNs (use filename for NN_load)
    NN_load(layer->attention->Q_proj, filename);
    NN_load(layer->attention->K_proj, filename);
    NN_load(layer->attention->V_proj, filename);
    NN_load(layer->attention->O_proj, filename);

    NN_load(layer->feed_forward->network, filename);

    NN_load(layer->norm1->norm_network, filename);
    NN_load(layer->norm2->norm_network, filename);

    transformer->layers[i] = layer;
  }

  fclose(file);
  return transformer;
}
