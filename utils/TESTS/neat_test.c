#include <stdio.h>
#include <stdlib.h>
#include "../NN/NEAT.h"

#define NUM_INPUTS 3
#define NUM_OUTPUTS 2
#define NUM_TESTS 5

void print_network_outputs(long double *outputs, size_t size) {
    if (!outputs) {
        printf("NULL outputs\n");
        return;
    }
    for (size_t i = 0; i < size; i++) {
        printf("%Lf ", outputs[i]);
    }
    printf("\n");
}

int main() {
    printf("Testing NEAT Implementation\n");
    printf("==========================\n\n");

    // Initialize random seed
    srand(42);

    // Test 1: Initialize NEAT network
    printf("Test 1: Initialize NEAT network\n");
    NEAT_t *neat = NEAT_init(NUM_INPUTS, NUM_OUTPUTS);
    if (!neat) {
        printf("FAILED: Could not initialize NEAT network\n");
        return 1;
    }
    printf("PASSED: NEAT network initialized\n\n");

    // Test 2: Add random nodes
    printf("Test 2: Add random nodes\n");
    NEAT_add_random(neat, 2);
    if (neat->num_nodes != 3) {
        printf("FAILED: Expected 3 nodes, got %u\n", neat->num_nodes);
        NEAT_destroy(neat);
        return 1;
    }
    printf("PASSED: Random nodes added successfully\n\n");

    // Test 3: Forward pass with valid inputs
    printf("Test 3: Forward pass with valid inputs\n");
    long double test_inputs[NUM_INPUTS] = {0.5, -0.3, 0.8};
    printf("\nRunning forward pass...\n");
    long double* output = NEAT_forward(neat, test_inputs);
    if (!output) {
        fprintf(stderr, "Forward pass failed\n");
        NEAT_destroy(neat);
        return 1;
    }

    printf("Input: ");
    print_network_outputs(test_inputs, NUM_INPUTS);
    printf("Output: [");
    for (size_t i = 0; i < NUM_OUTPUTS; i++) {
        printf("%.6Lf%s", output[i], i < NUM_OUTPUTS - 1 ? ", " : "");
    }
    printf("]\n");

    printf("PASSED: Forward pass produced output\n\n");

    // Test 4: Crossover
    printf("Test 4: Crossover test\n");
    NEAT_t *neat2 = NEAT_init(NUM_INPUTS, NUM_OUTPUTS);
    if (!neat2) {
        printf("FAILED: Could not initialize second NEAT network\n");
        NEAT_destroy(neat);
        return 1;
    }

    // Add random nodes to second network
    NEAT_add_random(neat2, 2);

    // Set different fitness values
    neat->nodes[0]->fitness = 0.8;
    neat2->nodes[0]->fitness = 0.6;

    Perceptron_t *offspring = NEAT_crossover(neat->nodes[0], neat2->nodes[0]);
    if (!offspring) {
        printf("FAILED: Crossover returned NULL\n");
        NEAT_destroy(neat);
        NEAT_destroy(neat2);
        return 1;
    }

    // Test offspring with same inputs
    long double *offspring_outputs = NN_forward(offspring->nn, test_inputs);
    printf("Offspring output: ");
    print_network_outputs(offspring_outputs, NUM_OUTPUTS);
    printf("PASSED: Crossover produced valid offspring\n\n");

    // Test 5: Multiple forward passes
    printf("Test 5: Multiple forward passes\n");
    for (int i = 0; i < NUM_TESTS; i++) {
        long double random_inputs[NUM_INPUTS];
        for (int j = 0; j < NUM_INPUTS; j++) {
            random_inputs[j] = ((double)rand() / RAND_MAX) * 2 - 1; // Random values between -1 and 1
        }
        printf("\nRunning forward pass...\n");
        long double* test_outputs = NEAT_forward(neat, random_inputs);
        if (!test_outputs) {
            fprintf(stderr, "Forward pass failed\n");
            NEAT_destroy(neat);
            return 1;
        }

        printf("Test %d:\n", i + 1);
        printf("Input: ");
        print_network_outputs(random_inputs, NUM_INPUTS);
        printf("Output: [");
        for (size_t i = 0; i < NUM_OUTPUTS; i++) {
            printf("%.6Lf%s", test_outputs[i], i < NUM_OUTPUTS - 1 ? ", " : "");
        }
        printf("]\n");
        free(test_outputs);
    }
    printf("PASSED: Multiple forward passes completed\n\n");

    // Test 6: Backpropagation
    printf("Test 6: Backpropagation test\n");
    printf("\nRunning backpropagation...\n");
    long double y_true = 1.0L;  // Target output
    NEAT_backprop(neat, test_inputs, y_true, output[0]);
    printf("PASSED: Backpropagation completed\n\n");

    // Clean up
    printf("Test 7: Memory cleanup\n");
    Perceptron_destroy(offspring);
    NEAT_destroy(neat2);
    NEAT_destroy(neat);
    printf("PASSED: Memory cleanup completed\n\n");

    printf("All tests completed successfully!\n");
    return 0;
}
