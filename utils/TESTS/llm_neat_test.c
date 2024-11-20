#include "../NN/LLM_NEAT.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Simple tokenizer implementation
int* tokenize(const char* text) {
    size_t len = strlen(text);
    int* tokens = (int*)malloc((len + 1) * sizeof(int));
    
    for(size_t i = 0; i < len; i++) {
        tokens[i] = (int)text[i];
    }
    tokens[len] = -1; // End token
    
    return tokens;
}

void test_tokenizer() {
    // Test cases with different lengths and letters
    char *test_cases[] = {
        "cat",           // c:3, a:1, t:20
        "bat",          // b:2, a:1, t:20
        "dog",          // d:4, o:15, g:7
        "zebra",        // z:26, e:5, b:2, r:18, a:1
        "programming",  // p:16, r:18, o:15, g:7, r:18, a:1, m:13, m:13, i:9, n:14, g:7
        "a",            // Single letter
        "aaa",          // Same letter repeated
        "xyz"           // End of alphabet
    };
    
    for (int i = 0; i < sizeof(test_cases)/sizeof(test_cases[0]); i++) {
        printf("\nTesting word: %s\n", test_cases[i]);
        int *tokens = tokenize(test_cases[i]);
        
        if (tokens == NULL) {
            printf("Tokenizer returned NULL\n");
            continue;
        }
        
        printf("Tokens: ");
        int j = 0;
        while (tokens[j] != -1) {
            printf("%d ", tokens[j]);
            j++;
        }
        printf("\n");
        
        free(tokens);
    }
}

void test_neat_transformer() {
    printf("\nTesting NEAT Transformer...\n");
    
    // Initialize NEAT transformer with small dimensions for testing
    size_t input_dim = 10;
    size_t num_heads = 2;
    size_t ff_dim = 20;
    
    NEAT_Transformer* transformer = neat_transformer_init(input_dim, num_heads, ff_dim);
    
    // Create test input
    long double input[10] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    long double target[10] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    long double learning_rate = 0.01;
    
    // Forward pass
    printf("\nRunning forward pass...\n");
    long double* output = llm_neat_forward(transformer, input);
    if (!output) {
        fprintf(stderr, "Forward pass failed\n");
        free(input);
        free(target);
        free_neat_transformer(transformer);
        return 1;
    }

    // Print output
    printf("Output: [");
    for (size_t i = 0; i < input_dim; i++) {
        printf("%.6Lf%s", output[i], i < input_dim - 1 ? ", " : "");
    }
    printf("]\n");

    // Backpropagation
    printf("\nRunning backpropagation...\n");
    long double* grad_output = malloc(input_dim * sizeof(long double));
    long double* grad_input = malloc(input_dim * sizeof(long double));
    for (size_t i = 0; i < input_dim; i++) {
        grad_output[i] = (output[i] - target[i]) / input_dim;  // Simple gradient
    }

    llm_neat_backprop(transformer, input, grad_output, grad_input);
    llm_neat_update(transformer, learning_rate);
    
    // Clean up
    free(output);
    free(grad_output);
    free(grad_input);
    neat_transformer_destroy(transformer);
}

int main() {
    test_tokenizer();
    test_neat_transformer();
    return 0;
}
