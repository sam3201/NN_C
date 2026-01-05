#ifndef NEAT_H
#define NEAT_H

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Node types
typedef enum { INPUT_NODE, HIDDEN_NODE, OUTPUT_NODE, BIAS_NODE } NodeType;

// Activation functions
typedef enum { SIGMOID = 0, TANH, RELU, LINEAR } ActivationFunctionType;

// Node structure
typedef struct Node {
  size_t id;
  NodeType type;
  ActivationFunctionType actFunc;
  long double value; // current activation
} Node;

// Connection structure
typedef struct Connection {
  size_t innovation; // innovation number
  Node *from;
  Node *to;
  long double weight;
  bool enabled;
} Connection;

// Genome structure (a single NEAT network)
typedef struct Genome {
  size_t numNodes;
  Node **nodes;

  size_t numConnections;
  Connection **connections;

  long double fitness;
} Genome;

// ------------------- Constructors / Destructors -------------------
Genome *Genome_init(size_t numInputs, size_t numOutputs);
void Genome_destroy(Genome *genome);

// ------------------- NEAT Operators -------------------
void Genome_add_node(Genome *genome, size_t connectionIndex);
void Genome_add_connection(Genome *genome, size_t fromNode, size_t toNode,
                           long double weight);
void Genome_mutate_weights(Genome *genome, long double perturbRate,
                           long double perturbAmount);
Genome *Genome_crossover(Genome *parent1, Genome *parent2);

// ------------------- Forward Propagation -------------------
void Genome_forward(Genome *genome, long double *input, long double *output);
long double activate(long double x, ActivationFunctionType type);

// ------------------- Serialization -------------------
void Genome_save(Genome *genome, const char *filename);
Genome *Genome_load(const char *filename);

size_t *topological_sort(Genome *genome, size_t *outSize);

#endif // NEAT_H
