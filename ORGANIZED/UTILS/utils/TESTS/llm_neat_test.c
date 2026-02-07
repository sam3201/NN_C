#include "../NN/NEAT.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define INPUT_DIM 64
#define MODEL_DIM 256
#define NUM_HEADS 8
#define POPULATION_SIZE 10
#define NUM_GENERATIONS 100
#define MUTATION_RATE 0.1L

// Generate random data for testing
void generate_random_data(long double* data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        data[i] = ((long double)rand() / RAND_MAX) * 2.0L - 1.0L;
    }
}

// Test evolution capabilities
void test_evolution() {
    printf("Testing evolution...\n");
    
    // Initialize a single NEAT population
    NEAT_t * neat = NEAT_init(INPUT_DIM, MODEL_DIM, POPULATION_SIZE);
    if (!neat) {
        printf("Failed to create NEAT population\n");
            return;
    }

    // Generate test data
    long double input[INPUT_DIM];
    long double target[MODEL_DIM];
    generate_random_data(input, INPUT_DIM);
    generate_random_data(target, MODEL_DIM);

    // Evolution loop
    for (size_t gen = 0; gen < NUM_GENERATIONS; gen++) {
        // Calculate fitness for each perceptron in the population
        for (unsigned int i = 0; i < neat->num_nodes; i++) {
            if (neat->nodes[i] && neat->nodes[i]->enabled) {
                long double* output = NEAT_forward(neat, input);
                if (output) {
                    // Calculate MSE loss
                    long double loss = 0.0L;
                    for (size_t j = 0; j < MODEL_DIM; j++) {
                        long double diff = output[j] - target[j];
                        loss += diff * diff;
                    }
                    loss /= MODEL_DIM;
                    neat->nodes[i]->fitness = -loss;  // Negative because higher fitness is better
                    free(output);
                }
            }
        }

        // Evolve the population
        NEAT_evolve(neat);
        
        if ((gen + 1) % 10 == 0) {
            printf("Generation %zu completed\n", gen + 1);
        }
    }

    // Cleanup
    NEAT_destroy(neat);
}

// Test dynamic scaling capabilities
void test_dynamic_scaling() {
    printf("Testing dynamic scaling...\n");
    
    // Create a NEAT population
    NEAT_t * neat = NEAT_init(INPUT_DIM, MODEL_DIM, POPULATION_SIZE);
    if (!neat) {
        printf("Failed to create NEAT population\n");
        return;
    }

    // Generate test data
    long double input[INPUT_DIM];
    long double target[MODEL_DIM];
    generate_random_data(input, INPUT_DIM);
    generate_random_data(target, MODEL_DIM);

    // Test training and evolution
    for (size_t i = 0; i < 10; i++) {
        NEAT_train(neat, input, target);
        NEAT_evolve(neat);
        
        // Add random neurons to test dynamic scaling
        if (i % 3 == 0) {
            NEAT_add_neuron_random(neat);
        }
    }

    // Cleanup
    NEAT_destroy(neat);
}

int main() {
    // Initialize random seed
    srand(time(NULL));

    // Run tests
    test_evolution();
    test_dynamic_scaling();

    printf("Test completed\n");
    return 0;
}
