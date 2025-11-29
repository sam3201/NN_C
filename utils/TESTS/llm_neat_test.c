#include "../NN/NEAT.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
    
    // Initialize population
    NEAT_t * population[POPULATION_SIZE];
    for (size_t i = 0; i < POPULATION_SIZE; i++) {
        population[i] = NEAT_init(INPUT_DIM, MODEL_DIM, NUM_HEADS);
        if (!population[i]) {
            printf("Failed to create transformer %zu\n", i);
            return;
        }
    }

    // Generate test data
    long double input[INPUT_DIM];
    long double target[INPUT_DIM];
    generate_random_data(input, INPUT_DIM);
    generate_random_data(target, INPUT_DIM);

    // Evolution loop
    for (size_t gen = 0; gen < NUM_GENERATIONS; gen++) {
        // Calculate fitness for each transformer
        long double fitness[POPULATION_SIZE];
        for (size_t i = 0; i < POPULATION_SIZE; i++) {
            fitness[i] = calculate_fitness(population[i], input, target);
        }

        // Create next generation through crossover and mutation
        for (size_t i = 0; i < POPULATION_SIZE / 2; i++) {
            size_t parent1_idx = rand() % POPULATION_SIZE;
            size_t parent2_idx = rand() % POPULATION_SIZE;
            
            NEAT_Transformer* child = neat_crossover(population[parent1_idx], population[parent2_idx]);
            if (child) {
                neat_mutate(child, MUTATION_RATE);
                free_neat_transformer(population[i]);
                population[i] = child;
            }
        }
    }

    // Cleanup
    for (size_t i = 0; i < POPULATION_SIZE; i++) {
        free_neat_transformer(population[i]);
    }
}

// Test dynamic scaling capabilities
void test_dynamic_scaling() {
    printf("Testing dynamic scaling...\n");
    
    // Create a transformer
    NEAT_Transformer* transformer = create_neat_transformer(INPUT_DIM, MODEL_DIM, NUM_HEADS);
    if (!transformer) {
        printf("Failed to create transformer\n");
        return;
    }

    // Generate test data
    long double input[INPUT_DIM];
    long double target[INPUT_DIM];
    generate_random_data(input, INPUT_DIM);
    generate_random_data(target, INPUT_DIM);

    // Test architecture optimization
    optimize_architecture(transformer, input, target);

    // Cleanup
    free_neat_transformer(transformer);
}

// Calculate fitness for a given transformer
long double calculate_fitness(NEAT_Transformer* transformer, long double* input, long double* target) {
    if (!transformer || !input || !target) return INFINITY;

    // Forward pass
    long double* output = llm_neat_forward(transformer, input);
    if (!output) return INFINITY;

    // Calculate MSE loss
    long double loss = 0.0L;
    size_t output_size = transformer->input_dim;  // Using input_dim as output_dim for now
    for (size_t i = 0; i < output_size; i++) {
        long double diff = output[i] - target[i];
        loss += diff * diff;
    }
    loss /= output_size;

    free(output);
    return -loss;  // Negative because higher fitness is better
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
