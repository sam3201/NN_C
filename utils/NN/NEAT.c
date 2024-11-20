#include "NEAT.h"
#include "NN.h"
#include "../VISUALIZER/NN_visualizer.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>

Perceptron_t *Perceptron_init(NN_t *nn, long double fitness, unsigned int num_connections, Perceptron_t *connections, bool enabled) {
    if (!nn) {
        fprintf(stderr, "Neural network cannot be NULL\n");
        return NULL;
    }

    Perceptron_t *perceptron = malloc(sizeof(Perceptron_t));
    if (!perceptron) {
        fprintf(stderr, "Failed to allocate memory for perceptron\n");
        return NULL;
    }

    perceptron->nn = nn;
    perceptron->fitness = fitness;
    perceptron->num_connections = num_connections;
    perceptron->connections = connections;
    perceptron->enabled = enabled;

    return perceptron;
}

Perceptron_t *Perceptron_init_random(unsigned int num_inputs, unsigned int num_outputs) {
    if (num_inputs == 0 || num_outputs == 0) {
        fprintf(stderr, "Invalid input/output dimensions\n");
        return NULL;
    }

    NN_t *nn = NN_init_random(num_inputs, num_outputs); 
    if (!nn) {
        fprintf(stderr, "Failed to initialize neural network\n");
        return NULL;
    }
  
    Perceptron_t *perceptron = Perceptron_init(nn, 0.0, 0, NULL, true);
    if (!perceptron) {
        NN_destroy(nn);
        return NULL;
    }

    return perceptron;
}

void Perceptron_destroy(Perceptron_t *perceptron) {
    if (perceptron) {
        if (perceptron->nn) {
            NN_destroy(perceptron->nn);
        }
        free(perceptron);
    }
}

NEAT_t *NEAT_init(unsigned int num_inputs, unsigned int num_outputs) {
    if (num_inputs == 0 || num_outputs == 0) {
        fprintf(stderr, "Invalid input/output dimensions\n");
        return NULL;
    }

    NEAT_t *neat = malloc(sizeof(NEAT_t));
    if (!neat) {
        fprintf(stderr, "Failed to allocate memory for NEAT structure\n");
        return NULL;
    }

    // Initialize all fields to 0/NULL
    neat->num_nodes = 0;
    neat->nodes = NULL;
    neat->species_id = 0;

    // Initialize with one node
    neat->num_nodes = 1;
    neat->nodes = calloc(neat->num_nodes, sizeof(Perceptron_t *));
    if (!neat->nodes) {
        fprintf(stderr, "Failed to allocate memory for nodes\n");
        free(neat);
        return NULL;
    }

    // Create initial perceptron
    neat->nodes[0] = Perceptron_init_random(num_inputs, num_outputs);
    if (!neat->nodes[0]) {
        fprintf(stderr, "Failed to initialize perceptron\n");
        free(neat->nodes);
        free(neat);
        return NULL;
    }

    return neat;
}

void NEAT_add_random(NEAT_t *neat, unsigned int num_new_nodes) {
    if (!neat || num_new_nodes == 0) {
        fprintf(stderr, "Invalid NEAT structure or number of nodes\n");
        return;
    }

    // Get input/output dimensions from existing node
    if (!neat->nodes || !neat->nodes[0] || !neat->nodes[0]->nn) {
        fprintf(stderr, "Invalid NEAT structure: no valid initial node\n");
        return;
    }

    unsigned int num_inputs = neat->nodes[0]->nn->layers[0];
    unsigned int num_outputs = neat->nodes[0]->nn->layers[neat->nodes[0]->nn->numLayers - 1];

    // Allocate new array with increased size
    Perceptron_t **new_nodes = realloc(neat->nodes, (neat->num_nodes + num_new_nodes) * sizeof(Perceptron_t *));
    if (!new_nodes) {
        fprintf(stderr, "Failed to reallocate memory for new nodes\n");
        return;
    }
    neat->nodes = new_nodes;

    // Initialize new nodes
    for (unsigned int i = 0; i < num_new_nodes; i++) {
        neat->nodes[neat->num_nodes + i] = Perceptron_init_random(num_inputs, num_outputs);
        if (!neat->nodes[neat->num_nodes + i]) {
            fprintf(stderr, "Failed to initialize node %u\n", neat->num_nodes + i);
            continue;
        }
        
        // Randomly set species ID (for now, just increment)
        neat->species_id++;
        
        printf("Added new node %u with species ID %d\n", neat->num_nodes + i, neat->species_id);
    }

    neat->num_nodes += num_new_nodes;
    printf("NEAT network now has %u nodes\n", neat->num_nodes);
}

void NEAT_destroy(NEAT_t *neat) {
    printf("\nDestroying NEAT network...\n");
    if (!neat) {
        printf("NEAT is NULL, nothing to destroy\n");
        return;
    }

    printf("Destroying nodes...\n");
    if (neat->nodes) {
        for (int i = 0; i < neat->num_nodes; i++) {
            if (neat->nodes[i]) {
                printf("Destroying node %d...\n", i);
                if (neat->nodes[i]->nn) {
                    printf("Destroying neural network for node %d...\n", i);
                    NN_destroy(neat->nodes[i]->nn);
                    neat->nodes[i]->nn = NULL;
                }
                NEAT_destroy_node(neat->nodes[i]);
                neat->nodes[i] = NULL;
            }
        }
        printf("Freeing nodes array...\n");
        free(neat->nodes);
        neat->nodes = NULL;
    }

    printf("Freeing NEAT structure...\n");
    free(neat);
    printf("NEAT network destroyed successfully\n");
}

void NEAT_destroy_node(Perceptron_t *node) {
    printf("Destroying node...\n");
    if (!node) {
        printf("Node is NULL, nothing to destroy\n");
        return;
    }

    if (node->nn) {
        printf("Destroying node's neural network...\n");
        NN_destroy(node->nn);
        node->nn = NULL;
    }

    printf("Freeing node structure...\n");
    free(node);
    printf("Node destroyed successfully\n");
}

void NEAT_add_neuron_random(NEAT_t *neat) {
    if (!neat || !neat->nodes || neat->num_nodes == 0) {
        fprintf(stderr, "Invalid NEAT structure for adding neuron\n");
        return;
    }

    // Expand nodes array
    neat->num_nodes++;
    Perceptron_t **new_nodes = realloc(neat->nodes, neat->num_nodes * sizeof(Perceptron_t *));
    if (!new_nodes) {
        fprintf(stderr, "Failed to reallocate memory for nodes\n");
        neat->num_nodes--; // Revert the increment
        return;
    }
    neat->nodes = new_nodes;

    // Get a random existing node as template
    unsigned int rand_idx = rand() % (neat->num_nodes - 1);
    Perceptron_t *template_node = neat->nodes[rand_idx];

    if (!template_node || !template_node->nn) {
        fprintf(stderr, "Invalid template node\n");
        neat->num_nodes--;
        return;
    }

    // Create a new node with random layer sizes between template's bounds
    unsigned int min_layer_size = 1;
    unsigned int max_layer_size = 10;
    unsigned int num_hidden_layers = (rand() % 3) + 1; // 1 to 3 hidden layers
    
    // Initialize layer sizes array
    size_t *layer_sizes = malloc((num_hidden_layers + 2) * sizeof(size_t)); // +2 for input and output layers
    if (!layer_sizes) {
        fprintf(stderr, "Failed to allocate layer sizes\n");
        neat->num_nodes--;
        return;
    }
    
    // Set input and output layer sizes
    layer_sizes[0] = template_node->nn->layers[0]; // Input layer
    layer_sizes[num_hidden_layers + 1] = template_node->nn->layers[template_node->nn->numLayers - 1]; // Output layer
    
    // Generate random hidden layer sizes
    for (unsigned int i = 1; i <= num_hidden_layers; i++) {
        layer_sizes[i] = min_layer_size + (rand() % (max_layer_size - min_layer_size + 1));
    }

    // Initialize activation functions and derivatives
    ActivationFunctionType *activation_funcs = malloc((num_hidden_layers + 2) * sizeof(ActivationFunctionType));
    ActivationDerivativeType *activation_derivs = malloc((num_hidden_layers + 2) * sizeof(ActivationDerivativeType));
    
    if (!activation_funcs || !activation_derivs) {
        fprintf(stderr, "Failed to allocate activation functions\n");
        free(layer_sizes);
        if (activation_funcs) free(activation_funcs);
        if (activation_derivs) free(activation_derivs);
        neat->num_nodes--;
        return;
    }

    // Assign activation functions with preference for stable ones
    for (unsigned int i = 0; i < num_hidden_layers + 2; i++) {
        int r = rand() % 100;
        if (r < 40) { // 40% chance for ReLU
            activation_funcs[i] = RELU;
            activation_derivs[i] = RELU_DERIVATIVE;
        } else if (r < 70) { // 30% chance for tanh
            activation_funcs[i] = TANH;
            activation_derivs[i] = TANH_DERIVATIVE;
        } else { // 30% chance for sigmoid
            activation_funcs[i] = SIGMOID;
            activation_derivs[i] = SIGMOID_DERIVATIVE;
        }
    }

    // Create new neural network with proper initialization
    NN_t *new_nn = NN_init(layer_sizes, 
                          activation_funcs,
                          activation_derivs,
                          MSE,  // Use Mean Squared Error as loss function
                          MSE_DERIVATIVE,
                          0.01); // Learning rate

    if (!new_nn) {
        fprintf(stderr, "Failed to create new neural network\n");
        free(layer_sizes);
        free(activation_funcs);
        free(activation_derivs);
        neat->num_nodes--;
        return;
    }

    // Initialize the new perceptron
    neat->nodes[neat->num_nodes - 1] = malloc(sizeof(Perceptron_t));
    if (!neat->nodes[neat->num_nodes - 1]) {
        fprintf(stderr, "Failed to allocate new perceptron\n");
        NN_destroy(new_nn);
        free(layer_sizes);
        free(activation_funcs);
        free(activation_derivs);
        neat->num_nodes--;
        return;
    }

    neat->nodes[neat->num_nodes - 1]->nn = new_nn;
    neat->nodes[neat->num_nodes - 1]->enabled = true;
    neat->nodes[neat->num_nodes - 1]->fitness = 0.0f;

    // Initialize weights with Xavier initialization
    for (unsigned int i = 0; i < new_nn->numLayers - 1; i++) {
        double scale = sqrt(2.0 / (new_nn->layers[i] + new_nn->layers[i + 1]));
        for (unsigned int j = 0; j < new_nn->layers[i + 1] * new_nn->layers[i]; j++) {
            // Generate random number between -1 and 1
            double rand_val = ((double)rand() / RAND_MAX) * 2 - 1;
            new_nn->weights[i][j] = rand_val * scale;
        }
        
        // Initialize biases to small random values
        for (unsigned int j = 0; j < new_nn->layers[i + 1]; j++) {
            new_nn->biases[i][j] = ((double)rand() / RAND_MAX) * 0.2 - 0.1;
        }
    }

    free(layer_sizes);
    free(activation_funcs);
    free(activation_derivs);
}

Perceptron_t *NEAT_crossover(Perceptron_t *parent1, Perceptron_t *parent2) {
    printf("\nStarting NEAT crossover...\n");
    
    // Validate parents
    if (!parent1 || !parent2) {
        fprintf(stderr, "Invalid parent nodes for crossover\n");
        return NULL;
    }
    
    if (!parent1->nn || !parent2->nn) {
        fprintf(stderr, "Parent neural networks are NULL\n");
        return NULL;
    }

    // Validate parent network structures
    if (parent1->nn->layers[0] != parent2->nn->layers[0] ||
        parent1->nn->layers[parent1->nn->numLayers - 1] != parent2->nn->layers[parent2->nn->numLayers - 1]) {
        fprintf(stderr, "Parent networks have incompatible input/output dimensions\n");
        return NULL;
    }
    
    printf("Parent 1 network: Layers=%zu, Input=%zu, Output=%zu\n",
           parent1->nn->numLayers,
           parent1->nn->layers[0],
           parent1->nn->layers[parent1->nn->numLayers - 1]);
    printf("Parent 2 network: Layers=%zu, Input=%zu, Output=%zu\n",
           parent2->nn->numLayers,
           parent2->nn->layers[0],
           parent2->nn->layers[parent2->nn->numLayers - 1]);

    // Create offspring with structure based on fitness-weighted average of parents
    Perceptron_t *offspring = malloc(sizeof(Perceptron_t));
    if (!offspring) {
        fprintf(stderr, "Failed to allocate memory for offspring\n");
        return NULL;
    }

    // Initialize offspring fields to prevent undefined behavior
    offspring->nn = NULL;
    offspring->enabled = true;
    offspring->fitness = 0;
    offspring->num_connections = 0;
    offspring->connections = NULL;

    // Determine better and weaker parent based on fitness
    Perceptron_t *better_parent = (parent1->fitness >= parent2->fitness) ? parent1 : parent2;
    Perceptron_t *weaker_parent = (parent1->fitness >= parent2->fitness) ? parent2 : parent1;

    // Initialize layer sizes array with zero terminator
    size_t num_layers = better_parent->nn->numLayers;
    size_t *layer_sizes = calloc(num_layers + 1, sizeof(size_t)); // +1 for zero terminator
    if (!layer_sizes) {
        fprintf(stderr, "Failed to allocate memory for layer sizes\n");
        free(offspring);
        return NULL;
    }

    // Copy layer sizes with potential mutations
    layer_sizes[0] = better_parent->nn->layers[0]; // Input layer size (fixed)
    layer_sizes[num_layers - 1] = better_parent->nn->layers[num_layers - 1]; // Output layer size (fixed)
    for (unsigned int i = 1; i < num_layers - 1; i++) {
        if (rand() % 100 < 10) {
            // 10% chance to mutate hidden layer size
            int size_diff = (rand() % 3) - 1; // -1, 0, or 1
            layer_sizes[i] = better_parent->nn->layers[i] + size_diff;
            if (layer_sizes[i] < 1) layer_sizes[i] = 1;
            if (layer_sizes[i] > 10) layer_sizes[i] = 10;
        } else {
            layer_sizes[i] = better_parent->nn->layers[i];
        }
    }
    layer_sizes[num_layers] = 0; // Zero terminator

    // Initialize activation functions and derivatives
    ActivationFunctionType *activation_funcs = malloc(num_layers * sizeof(ActivationFunctionType));
    ActivationDerivativeType *activation_derivs = malloc(num_layers * sizeof(ActivationDerivativeType));
    
    if (!activation_funcs || !activation_derivs) {
        fprintf(stderr, "Failed to allocate memory for activation functions\n");
        free(layer_sizes);
        if (activation_funcs) free(activation_funcs);
        if (activation_derivs) free(activation_derivs);
        free(offspring);
        return NULL;
    }

    // Crossover activation functions with mutation
    for (unsigned int i = 0; i < num_layers; i++) {
        // Initialize with better parent's activation functions
        activation_funcs[i] = activation_function_to_enum(better_parent->nn->activationFunctions[i]);
        activation_derivs[i] = activation_derivative_to_enum(better_parent->nn->activationDerivatives[i]);

        // 10% chance to mutate activation function (except for output layer)
        if (i < num_layers - 1 && rand() % 100 < 10) {
            int r = rand() % 3;
            switch (r) {
                case 0:
                    activation_funcs[i] = RELU;
                    activation_derivs[i] = RELU_DERIVATIVE;
                    break;
                case 1:
                    activation_funcs[i] = TANH;
                    activation_derivs[i] = TANH_DERIVATIVE;
                    break;
                case 2:
                    activation_funcs[i] = SIGMOID;
                    activation_derivs[i] = SIGMOID_DERIVATIVE;
                    break;
            }
        }
    }

    // Force sigmoid activation for output layer
    activation_funcs[num_layers - 1] = SIGMOID;
    activation_derivs[num_layers - 1] = SIGMOID_DERIVATIVE;

    // Create new neural network with MSE loss function
    offspring->nn = NN_init(layer_sizes,
                           activation_funcs,
                           activation_derivs,
                           MSE,
                           MSE_DERIVATIVE,
                           (better_parent->nn->learningRate + weaker_parent->nn->learningRate) / 2.0);

    if (!offspring->nn) {
        fprintf(stderr, "Failed to create offspring neural network\n");
        free(layer_sizes);
        free(activation_funcs);
        free(activation_derivs);
        free(offspring);
        return NULL;
    }

    // Perform crossover of weights and biases
    for (unsigned int i = 0; i < offspring->nn->numLayers - 1; i++) {
        size_t num_weights = offspring->nn->layers[i + 1] * offspring->nn->layers[i];
        size_t num_biases = offspring->nn->layers[i + 1];
        
        // Validate parent layer dimensions
        if (i < better_parent->nn->numLayers - 1 && i < weaker_parent->nn->numLayers - 1) {
            // Crossover weights with Xavier initialization
            for (size_t j = 0; j < num_weights; j++) {
                if (rand() % 100 < 80) { // 80% inheritance from better parent
                    offspring->nn->weights[i][j] = better_parent->nn->weights[i][j];
                } else {
                    offspring->nn->weights[i][j] = weaker_parent->nn->weights[i][j];
                }
                
                // Apply small random mutation (10% chance)
                if (rand() % 100 < 10) {
                    long double scale = sqrt(2.0 / offspring->nn->layers[i]); // Xavier initialization
                    offspring->nn->weights[i][j] += ((rand() / (long double)RAND_MAX) * 2 - 1) * scale * 0.1;
                }
            }
            
            // Crossover biases
            for (size_t j = 0; j < num_biases; j++) {
                if (rand() % 100 < 80) { // 80% inheritance from better parent
                    offspring->nn->biases[i][j] = better_parent->nn->biases[i][j];
                } else {
                    offspring->nn->biases[i][j] = weaker_parent->nn->biases[i][j];
                }
                
                // Apply small random mutation (10% chance)
                if (rand() % 100 < 10) {
                    offspring->nn->biases[i][j] += ((rand() / (long double)RAND_MAX) * 2 - 1) * 0.1;
                }
            }
        }
    }

    printf("Crossover complete. Offspring network: Layers=%zu, Input=%zu, Output=%zu\n",
           offspring->nn->numLayers,
           offspring->nn->layers[0],
           offspring->nn->layers[offspring->nn->numLayers - 1]);
    
    // Clean up
    free(layer_sizes);
    free(activation_funcs);
    free(activation_derivs);

    return offspring;
}

void NEAT_evolve(NEAT_t *neat) {
    if (!neat || !neat->nodes || neat->num_nodes == 0) {
        return;
    }

    // Sort nodes by fitness
    for (unsigned int i = 0; i < neat->num_nodes; i++) {
        for (unsigned int j = i + 1; j < neat->num_nodes; j++) {
            if (neat->nodes[j]->fitness > neat->nodes[i]->fitness) {
                Perceptron_t *temp = neat->nodes[i];
                neat->nodes[i] = neat->nodes[j];
                neat->nodes[j] = temp;
            }
        }
    }

    // Keep the best performing node
    Perceptron_t *best = neat->nodes[0];
    
    // Get input/output dimensions from best network
    size_t num_inputs = best->nn->layers[0];
    size_t num_outputs = best->nn->layers[best->nn->numLayers - 1];
    
    // Create new generation
    for (unsigned int i = 1; i < neat->num_nodes; i++) {
        // Destroy old node
        NEAT_destroy_node(neat->nodes[i]);
        
        // Create new node through mutation of best performer
        neat->nodes[i] = Perceptron_init_random(num_inputs, num_outputs);
        
        // Random chance to add new neuron
        if (rand() % 100 < 10) { // 10% chance
            NEAT_add_neuron_random(neat);
        }
    }
}

long double *NEAT_forward(NEAT_t *neat, long double inputs[]) {
    if (!neat || !inputs) {
        return NULL;
    }

    // Allocate memory for output
    long double* output = malloc(sizeof(long double));
    if (!output) return NULL;
    *output = 0;

    // Forward pass through each node in the network
    for (size_t i = 0; i < neat->num_nodes; i++) {
        if (neat->nodes[i]->enabled) {
            long double* node_output = perceptron_forward(neat->nodes[i], inputs);
            if (node_output) {
                *output += *node_output;
                free(node_output);
            }
        }
    }

    // Apply sigmoid activation to final output
    *output = 1.0L / (1.0L + expl(-*output));
    return output;
}

void NEAT_backprop(NEAT_t *neat, long double inputs[], long double y_true, long double y_pred) {
    // Backpropagate through each node in reverse order
    for (int i = neat->num_nodes - 1; i >= 0; i--) {
        if (neat->nodes[i]->enabled) {
            perceptron_backprop(neat->nodes[i], inputs, y_true, y_pred);
        }
    }
}

long double *perceptron_forward(Perceptron_t *perceptron, long double inputs[]) {
    // Forward pass through the perceptron's neural network
    return NN_forward(perceptron->nn, inputs);
}

void perceptron_backprop(Perceptron_t *perceptron, long double inputs[], long double y_true, long double y_pred) {
    // Backpropagate through the perceptron's neural network
    NN_backprop(perceptron->nn, inputs, y_true, y_pred);
}

void* NEAT_RunVisualizer(void* arg) {
    NEAT_t* neat = (NEAT_t*)arg;
    
    // Initialize window on main thread
    dispatch_async(dispatch_get_main_queue(), ^{
        const int screenWidth = 800;
        const int screenHeight = 600;
        InitWindow(screenWidth, screenHeight, "Neural Network Visualizer");
        SetTargetFPS(60);
        
        while (!WindowShouldClose()) {
            BeginDrawing();
            ClearBackground(RAYWHITE);
            
            // Draw neural network visualization here
            DrawText("Neural Network Visualization", 190, 200, 20, LIGHTGRAY);
            
            EndDrawing();
        }
        
        CloseWindow();
    });
    
    return NULL;
}
