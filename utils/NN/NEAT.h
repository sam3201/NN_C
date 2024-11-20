#ifndef NEAT_H
#define NEAT_H

#include "NN.h"
#include <stdbool.h>
#include <dispatch/dispatch.h>
#include "../Raylib/raylib.h"

typedef struct Perceptron_t Perceptron_t;

typedef struct Perceptron_t {
    NN_t *nn;                       // Neural Network
    long double fitness;            // Fitness of the Perceptron
    unsigned int num_connections;    // Number of connections
    Perceptron_t *connections;      // Array of connections
    bool enabled;                   // Whether the Perceptron is enabled
} Perceptron_t;

typedef struct {
    int species_id;                 // Species ID
    unsigned int num_nodes;         // Number of nodes
    Perceptron_t **nodes;          // Array of pointers to nodes
} NEAT_t;

// Perceptron functions
Perceptron_t *Perceptron_init(NN_t *nn, long double fitness, unsigned int num_connections, Perceptron_t *connections, bool enabled);
Perceptron_t *Perceptron_init_random(unsigned int num_inputs, unsigned int num_outputs);
void Perceptron_destroy(Perceptron_t *perceptron);

// NEAT initialization and management
NEAT_t *NEAT_init(unsigned int num_inputs, unsigned int num_outputs);
void NEAT_add_random(NEAT_t *neat, unsigned int num_nodes);
void NEAT_destroy(NEAT_t *neat);
void NEAT_destroy_node(Perceptron_t *node);

// NEAT network modification
void NEAT_add_neuron(NEAT_t *neat, unsigned int to, unsigned int from);
void NEAT_add_neuron_random(NEAT_t *neat);
Perceptron_t *NEAT_crossover(Perceptron_t *parent1, Perceptron_t *parent2);

// NEAT evolution and speciation
long double NEAT_compatibility_distance(Perceptron_t *perceptron1, Perceptron_t *perceptron2);
void NEAT_speciate(NEAT_t *neat);
void NEAT_evolve(NEAT_t *neat);

// NEAT network operations
long double *NEAT_forward(NEAT_t *neat, long double inputs[]);
void NEAT_backprop(NEAT_t *neat, long double inputs[], long double y_true, long double y_pred);
void NEAT_evolve(NEAT_t *neat);

// Perceptron operations
long double *perceptron_forward(Perceptron_t *perceptron, long double inputs[]);
void perceptron_backprop(Perceptron_t *perceptron, long double inputs[], long double y_true, long double y_pred);

// Thread function for running visualizer
void* NEAT_RunVisualizer(void* arg);

#endif
