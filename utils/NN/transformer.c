#include "transformer.h"
#include "NN.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Creation functions
MultiHeadAttention* create_attention(size_t model_dim, size_t num_heads) {
    MultiHeadAttention* mha = malloc(sizeof(MultiHeadAttention));
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

    mha->Q_proj = NN_init(layers, activations, derivatives, MSE, MSE_DERIVATIVE, L1, SGD, 0.01L);
    mha->K_proj = NN_init(layers, activations, derivatives, MSE, MSE_DERIVATIVE, L1, SGD, 0.01L);
    mha->V_proj = NN_init(layers, activations, derivatives, MSE, MSE_DERIVATIVE, L1, SGD, 0.01L);
    mha->O_proj = NN_init(layers, activations, derivatives, MSE, MSE_DERIVATIVE, L1, SGD, 0.01L);

    if (!mha->Q_proj || !mha->K_proj || !mha->V_proj || !mha->O_proj) {
        free_attention(mha);
        return NULL;
    }

    return mha;
}

FeedForward* create_feed_forward(size_t input_dim, size_t hidden_dim) {
    FeedForward* ff = malloc(sizeof(FeedForward));
    if (!ff) {
        return NULL;
    }

    ff->input_dim = input_dim;
    ff->hidden_dim = hidden_dim;

    size_t layers[] = {input_dim, hidden_dim, input_dim, 0};
    ActivationFunctionType activations[] = {RELU, RELU, RELU};
    ActivationDerivativeType derivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE, RELU_DERIVATIVE};

    ff->network = NN_init(layers, activations, derivatives, MSE, MSE_DERIVATIVE, L1, SGD, 0.01L);
    if (!ff->network) {
        free(ff);
        return NULL;
    }

    return ff;
}

LayerNorm* create_layer_norm(size_t dim, long double epsilon) {
    LayerNorm* ln = malloc(sizeof(LayerNorm));
    if (!ln) {
        return NULL;
    }

    ln->dim = dim;
    ln->epsilon = epsilon;

    size_t layers[] = {dim, dim, 0};
    ActivationFunctionType activations[] = {RELU, RELU};
    ActivationDerivativeType derivatives[] = {RELU_DERIVATIVE, RELU_DERIVATIVE};

    ln->norm_network = NN_init(layers, activations, derivatives, MSE, MSE_DERIVATIVE, L1, SGD, 0.01L);
    if (!ln->norm_network) {
        free(ln);
        return NULL;
    }

    return ln;
}

TransformerLayer* create_transformer_layer(size_t model_dim, size_t num_heads, size_t ff_dim) {
    TransformerLayer* layer = malloc(sizeof(TransformerLayer));
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
long double *transformer_mha_forward(MultiHeadAttention *mha, long double *input, size_t seq_length) {
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
                scores[i * seq_length + j] += Q[i * mha->head_dim + k] * K[j * mha->head_dim + k];
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
    long double *attention_output = malloc(seq_length * mha->head_dim * sizeof(long double));
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
                attention_output[i * mha->head_dim + j] += scores[i * seq_length + k] * V[k * mha->head_dim + j];
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
    long double *attention_output = transformer_mha_forward(layer->attention, input, layer->seq_length);
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
    long double *ff_output = NN_forward(layer->feed_forward->network, norm1_output);
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
void transformer_mha_backprop(MultiHeadAttention *mha, long double *input, long double *grad_output, long double *grad_input) {
    if (!mha || !input || !grad_output || !grad_input) {
        return;
    }

    // Backpropagate through output projection
    NN_backprop(mha->O_proj, input, grad_output[0], grad_output[0]);

    // Backpropagate through Q, K, V projections
    NN_backprop(mha->Q_proj, input, grad_output[0], grad_output[0]);
    NN_backprop(mha->K_proj, input, grad_output[0], grad_output[0]);
    NN_backprop(mha->V_proj, input, grad_output[0], grad_output[0]);

    // Update gradients for input
    for (size_t i = 0; i < mha->model_dim; i++) {
        grad_input[i] = grad_output[i];
    }
}

void transformer_norm_backprop(LayerNorm *ln, long double *input, long double *grad_output, long double *grad_input) {
    if (!ln || !input || !grad_output || !grad_input) {
        return;
    }

    // Calculate mean of input
    long double mean = 0;
    for (size_t i = 0; i < ln->dim; i++) {
        mean += input[i];
    }
    mean /= ln->dim;

    // Calculate variance of input
    long double var = 0;
    for (size_t i = 0; i < ln->dim; i++) {
        var += (input[i] - mean) * (input[i] - mean);
    }
    var /= ln->dim;

    // Backpropagate through normalization
    for (size_t i = 0; i < ln->dim; i++) {
        grad_input[i] = grad_output[i] / sqrt(var + ln->epsilon);
    }

    // Backpropagate through learned transformation
    NN_backprop(ln->norm_network, input, grad_output[0], grad_input[0]);
}

void transformer_backprop(TransformerLayer *layer, long double *input, long double *grad_output, long double *grad_input) {
    if (!layer || !input || !grad_output || !grad_input) {
        return;
    }

    // Allocate temporary gradients
    long double *grad_ff = malloc(layer->model_dim * sizeof(long double));
    long double *grad_norm1 = malloc(layer->model_dim * sizeof(long double));
    long double *grad_attn = malloc(layer->model_dim * sizeof(long double));

    if (!grad_ff || !grad_norm1 || !grad_attn) {
        free(grad_ff);
        free(grad_norm1);
        free(grad_attn);
        return;
    }

    // Backpropagate through second normalization layer
    transformer_norm_backprop(layer->norm2, input, grad_output, grad_ff);

    // Backpropagate through feed-forward network
    NN_backprop(layer->feed_forward->network, input, grad_ff[0], grad_norm1[0]);

    // Backpropagate through first normalization layer
    transformer_norm_backprop(layer->norm1, input, grad_norm1, grad_attn);

    // Backpropagate through attention layer
    transformer_mha_backprop(layer->attention, input, grad_attn, grad_input);

    // Clean up
    free(grad_ff);
    free(grad_norm1);
    free(grad_attn);
}

// Memory management functions
void free_attention(MultiHeadAttention *mha) {
    if (mha) {
        if (mha->Q_proj) NN_destroy(mha->Q_proj);
        if (mha->K_proj) NN_destroy(mha->K_proj);
        if (mha->V_proj) NN_destroy(mha->V_proj);
        if (mha->O_proj) NN_destroy(mha->O_proj);
        free(mha);
    }
}

void free_feed_forward(FeedForward *ff) {
    if (ff) {
        if (ff->network) NN_destroy(ff->network);
        free(ff);
    }
}

void free_layer_norm(LayerNorm *ln) {
    if (ln) {
        if (ln->norm_network) NN_destroy(ln->norm_network);
        free(ln);
    }
}

void free_transformer_layer(TransformerLayer *layer) {
    if (layer) {
        if (layer->attention) free_attention(layer->attention);
        if (layer->norm1) free_layer_norm(layer->norm1);
        if (layer->feed_forward) free_feed_forward(layer->feed_forward);
        if (layer->norm2) free_layer_norm(layer->norm2);
        free(layer);
    }
}
