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
void transformer_layernorm_backprop(LayerNorm *ln, long double *grad_output) {
  if (!ln || !grad_output || !ln->input_cache)
    return;

  size_t D = ln->dim;
  long double *x = ln->input_cache;

  // Compute mean and variance
  long double mean = 0;
  for (size_t i = 0; i < D; i++)
    mean += x[i];
  mean /= D;

  long double var = 0;
  for (size_t i = 0; i < D; i++)
    var += (x[i] - mean) * (x[i] - mean);
  var /= D;

  long double stddev = sqrt(var + ln->epsilon);

  // Gradient w.r.t normalized input
  long double *grad_norm = malloc(D * sizeof(long double));
  for (size_t i = 0; i < D; i++)
    grad_norm[i] = grad_output[i] / stddev;

  // Gradient w.r.t variance
  long double dvar = 0;
  for (size_t i = 0; i < D; i++)
    dvar +=
        grad_output[i] * (x[i] - mean) * -0.5L / ((var + ln->epsilon) * stddev);

  // Gradient w.r.t mean
  long double dmean = 0;
  for (size_t i = 0; i < D; i++)
    dmean += grad_output[i] * -1.0L / stddev + dvar * -2.0L * (x[i] - mean) / D;

  // Gradient w.r.t input
  long double *grad_input = malloc(D * sizeof(long double));
  for (size_t i = 0; i < D; i++)
    grad_input[i] = grad_norm[i] + dvar * 2.0L * (x[i] - mean) / D + dmean / D;

  NN_backprop(ln->norm_network, x, grad_input, 0.01L);

  free(grad_input);
  free(grad_norm);

  free(ln->input_cache);
  ln->input_cache = NULL;
}

// Feed-forward backprop
void transformer_feedforward_backprop(FeedForward *ff,
                                      long double *grad_output) {
  if (!ff || !grad_output || !ff->input_cache)
    return;

  NN_backprop(ff->network, ff->input_cache, grad_output, 0.01L);

  free(ff->input_cache);
  ff->input_cache = NULL;
}

// Transformer backprop through layers
void TRANSFORMER_backprop(Transformer_t *transformer,
                          long double **input_sequence, size_t seq_length,
                          long double *grad_loss) {
  if (!transformer || !input_sequence || !grad_loss)
    return;

  size_t L = transformer->num_layers;
  size_t D = transformer->model_dim;

  // Forward pass to cache intermediate states
  long double *output =
      TRANSFORMER_forward(transformer, input_sequence, seq_length);
  if (!output)
    return;

  // Start gradient from loss
  long double *grad = malloc(D * sizeof(long double));
  memcpy(grad, grad_loss, D * sizeof(long double));

  // Backprop through layers in reverse
  for (ssize_t i = L - 1; i >= 0; i--) {
    TransformerLayer *layer = transformer->layers[i];
    if (!layer)
      continue;

    // Norm2 backprop
    transformer_layernorm_backprop(layer->norm2, grad);

    // Feedforward backprop
    transformer_feedforward_backprop(layer->feed_forward, grad);

    // Norm1 backprop
    transformer_layernorm_backprop(layer->norm1, grad);

    // MHA backprop
    transformer_mha_backprop(layer->attention, grad);
  }

  free(grad);
  free(output);
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
