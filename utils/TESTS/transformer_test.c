#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../NN/transformer.h"

void test_attention() {
    printf("Testing Multi-Head Attention...\n");
    
    size_t model_dim = 64;
    size_t num_heads = 4;
    size_t seq_length = 1;
    
    MultiHeadAttention* mha = create_attention(model_dim, num_heads);
    if (!mha) {
        fprintf(stderr, "Failed to create attention layer\n");
        return;
    }
    printf("Attention layer created successfully\n");
    
    long double* test_input = malloc(model_dim * sizeof(long double));
    if (!test_input) {
        fprintf(stderr, "Failed to allocate test input\n");
        free_attention(mha);
        return;
    }
    
    for (size_t i = 0; i < model_dim; i++) {
        test_input[i] = sin(i * 0.1);
    }
    
    printf("Running forward pass...\n");
    long double* output = transformer_mha_forward(mha, test_input, seq_length);
    if (!output) {
        fprintf(stderr, "Forward pass failed\n");
        free(test_input);
        free_attention(mha);
        return;
    }
    
    printf("Output: [");
    for (size_t i = 0; i < model_dim; i++) {
        printf("%.6Lf%s", output[i], i < model_dim - 1 ? ", " : "");
    }
    printf("]\n");
    
    free(test_input);
    free(output);
    free_attention(mha);
    printf("Test completed successfully\n\n");
}

void test_layer_norm() {
    printf("Testing Layer Normalization...\n");
    
    size_t dim = 64;
    long double epsilon = 1e-6;
    
    LayerNorm* ln = create_layer_norm(dim, epsilon);
    if (!ln) {
        fprintf(stderr, "Failed to create layer norm\n");
        return;
    }
    printf("Layer norm created successfully\n");
    
    long double* test_input = malloc(dim * sizeof(long double));
    if (!test_input) {
        fprintf(stderr, "Failed to allocate test input\n");
        free_layer_norm(ln);
        return;
    }
    
    for (size_t i = 0; i < dim; i++) {
        test_input[i] = sin(i * 0.1);
    }
    
    printf("Running forward pass...\n");
    long double* output = transformer_norm_forward(ln, test_input);
    if (!output) {
        fprintf(stderr, "Forward pass failed\n");
        free(test_input);
        free_layer_norm(ln);
        return;
    }
    
    printf("Output: [");
    for (size_t i = 0; i < dim; i++) {
        printf("%.6Lf%s", output[i], i < dim - 1 ? ", " : "");
    }
    printf("]\n");
    
    free(test_input);
    free(output);
    free_layer_norm(ln);
    printf("Test completed successfully\n\n");
}

void test_transformer() {
    printf("Testing Complete Transformer Layer...\n");
    
    size_t model_dim = 64;
    size_t num_heads = 4;
    size_t ff_dim = 256;
    long double learning_rate = 0.01L;
    
    TransformerLayer* layer = create_transformer_layer(model_dim, num_heads, ff_dim);
    if (!layer) {
        fprintf(stderr, "Failed to create transformer layer\n");
        return;
    }
    printf("Transformer Layer initialized successfully\n");
    
    long double* input = malloc(model_dim * sizeof(long double));
    if (!input) {
        fprintf(stderr, "Failed to allocate test input\n");
        free_transformer_layer(layer);
        return;
    }
    
    for (size_t i = 0; i < model_dim; i++) {
        input[i] = sin(i * 0.1);
    }
    
    long double* target = malloc(model_dim * sizeof(long double));
    for (size_t i = 0; i < model_dim; i++) {
        target[i] = sin(i * 0.1) + 1.0;
    }
    
    printf("Running forward pass...\n");
    long double* output = transformer_forward(layer, input);
    if (!output) {
        fprintf(stderr, "Forward pass failed\n");
        free(input);
        free(target);
        free_transformer_layer(layer);
        return;
    }
    
    printf("Output: [");
    for (size_t i = 0; i < model_dim; i++) {
        printf("%.6Lf%s", output[i], i < model_dim - 1 ? ", " : "");
    }
    printf("]\n");
    
    printf("Running backpropagation...\n");
    long double* grad_output = malloc(model_dim * sizeof(long double));
    long double* grad_input = malloc(model_dim * sizeof(long double));
    
    for (size_t i = 0; i < model_dim; i++) {
        grad_output[i] = (output[i] - target[i]) / model_dim;  // Simple gradient
    }
    
    transformer_backprop(layer, input, grad_output, grad_input);
    
    free(input);
    free(target);
    free(output);
    free(grad_output);
    free(grad_input);
    free_transformer_layer(layer);
    printf("Test completed successfully\n\n");
}

int main() {
    test_attention();
    test_layer_norm();
    test_transformer();
    return 0;
}
