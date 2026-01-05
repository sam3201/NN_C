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

// Global innovation tracking
typedef struct {
  size_t from;
  size_t to;
  size_t innovation;
} InnovationRecord;

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

// Genome_t structure (a single NEAT network)
typedef struct Genome_t {
  size_t numNodes;
  Node **nodes;

  size_t numConnections;
  Connection **connections;

  long double fitness;
} Genome_t;

typedef struct {
  Genome_t **genomes;
  size_t size;
} Population;

static InnovationRecord *innovationHistory = NULL;
static size_t innovationCount = 0;

// ------------------- Constructors / Destructors -------------------
Genome_t *GENOME_init(size_t numInputs, size_t numOutputs);
void GENOME_destroy(Genome_t *genome);

// ------------------- NEAT Operators -------------------
void GENOME_add_node(Genome_t *genome, size_t connectionIndex);
void GENOME_add_connection(Genome_t *genome, size_t fromNode, size_t toNode,
                           long double weight);

void GENOME_mutate_weights(Genome_t *genome, long double perturbRate,
                           long double perturbAmount);
Genome_t *GENOME_crossover(Genome_t *p1, Genome_t *p2);

// ------------------- Forward Propagation -------------------
void GENOME_forwward(Genome_t *genome, long double *input, long double *output);
long double activate(long double x, ActivationFunctionType type);

// ------------------- Serialization -------------------
void GENOME_save(Genome_t *genome, const char *filename);
Genome_t *GENOME_load(const char *filename);

// ------------------- Population-------------------
Population *POPULATION_init(size_t popSize, size_t numInputs,
                            size_t numOutputs);

// ------------------- Innovation Number -------------------
size_t get_innovation_number(size_t from, size_t to);

// ------------------- Topological Sort -------------------
size_t *topological_sort(Genome_t *genome, size_t *outSize);

#endif // NEAT_H
