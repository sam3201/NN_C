#ifndef NEAT_H
#define NEAT_H

#include "NN.h"
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Node types
typedef enum { INPUT_NODE, HIDDEN_NODE, OUTPUT_NODE, BIAS_NODE } NodeType;

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

  NN_t *nn;
  long double fitness;
} Genome_t;

typedef struct {
  Genome_t **genomes;
  size_t size;
} Population;

typedef struct {
  Population *pop;
  size_t input_dims;
  size_t output_dims;
} NEAT_t;

static InnovationRecord *innovationHistory;
static size_t innovationCount;

// ------------------- Constructors / Destructors -------------------
Genome_t *GENOME_init(size_t *layers, ActivationFunctionType *actFuncs,
                      ActivationDerivativeType *actDerivs,
                      LossFunctionType lossFunc, LossDerivativeType lossDeriv,
                      RegularizationType reg, OptimizerType opt,
                      long double learningRate, size_t numInputs);
void GENOME_destroy(Genome_t *genome);

// ------------------- NEAT Operators -------------------
void GENOME_add_node(Genome_t *genome, size_t connectionIndex);
void GENOME_add_connection(Genome_t *genome, size_t fromNode, size_t toNode,
                           long double weight);

void GENOME_mutate_weights(Genome_t *genome, long double perturbRate,
                           long double perturbAmount);

// ------------------- Extended Mutation -------------------
void GENOME_mutate(Genome_t *genome, long double weightPerturbRate,
                   long double weightPerturbAmount, long double addNodeRate,
                   long double addConnectionRate,
                   long double toggleConnectionRate,
                   long double biasPerturbRate, long double actFuncMutateRate);

Genome_t *GENOME_crossover(Genome_t *p1, Genome_t *p2);

// ------------------- Forward Propagation -------------------
void GENOME_forward(Genome_t *genome, long double *input, long double *output);
long double activate(long double x, ActivationFunctionType type);

NN_t *GENOME_compile_to_NN(Genome_t *genome);

// ------------------- Serialization -------------------
void GENOME_save(Genome_t *genome, const char *filename);
Genome_t *GENOME_load(const char *filename);

// ------------------- Population-------------------
Population *POPULATION_init(size_t popSize, size_t numInputs,
                            size_t numOutputs);
void POPULATION_destroy(Population *pop);

// ------------------- Innovation Number -------------------
size_t get_innovation_number(size_t from, size_t to);

// ------------------- Compatability Distance -------------------
long double compatibility_distance(Genome_t *g1, Genome_t *g2, long double c1,
                                   long double c2, long double c3);

void POPULATION_evolve(Population *pop);
size_t assign_species(Population *pop);
long double compatibility_distance(Genome_t *g1, Genome_t *g2, long double c1,
                                   long double c2, long double c3);

NEAT_t *NEAT_init(size_t input_dim, size_t output_dim, size_t pop_size);
void NEAT_destroy(NEAT_t *neat);
void NEAT_train(NEAT_t *neat, long double **inputs, long double **targets,
                size_t numSamples);
void NEAT_reset_innovations();
size_t assign_species(Population *pop);

// ------------------- Topological Sort -------------------
size_t *topological_sort(Genome_t *genome, size_t *outSize);

#endif // NEAT_H
